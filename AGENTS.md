# AGENTS.md

## Project

Repository: `fafb-crnn`

This project builds a **connectome-constrained recurrent neural network (CRNN)** from the FlyWire FAFB dataset: the Female Adult Fly Brain connectome of *Drosophila melanogaster*.

The long-term experiment is to test whether the fly-derived visual system and brain produce representations from which high-level visual categories can be decoded.

The intended pipeline is:

```text
image / frame sequence
    ↓
fly-inspired visual encoder
    ↓
identified FAFB visual input neurons
    ↓
FAFB recurrent network
    ↓
optic lobe + central brain
    ↓
descending neurons
    ↓
small decoder
    ↓
animal class
```

The scientific question is **not** “can a fly recognize cats?”. A more accurate framing is:

> Does the connectivity and visual-processing architecture of the fly brain produce representations from which high-level visual categories can be decoded?

Treat the project as a computational-neuroscience / ML experiment. Do not overclaim biological fidelity.

---

## Working style

This is a **terminal-first Python project**.

Do:

- use normal Python source files and CLI scripts
- use `argparse` for scripts that need options
- keep reusable logic under `src/`
- keep executable entry points under `scripts/`
- put every data-processing and table-inspection step in `scripts/` so it can be rerun
- write deterministic preprocessing
- track *what to do next* in GitHub issues; keep *how to build it* in this file
- add assertions and sanity checks
- save expensive processed graph artifacts for reuse
- use type hints where useful
- make biological assumptions explicit in comments/docs
- inspect actual FAFB taxonomy/data before writing filters against labels

Do not:

- add Jupyter notebooks
- leave processing in chat snippets, notebooks, or one-off `python -c` commands
- use dense 139k × 139k adjacency matrices
- put unrelated helpers into a generic `utils.py`
- reload and reprocess the full connectivity CSV during every training run
- invent taxonomy values that have not been verified in the data
- describe engineering approximations as biologically exact

---

## Repository layout

Preferred structure:

```text
fafb-crnn/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── fafb/
│   │   ├── raw/
│   │   └── processed/
│   └── animals10/
│       ├── raw/
│       └── processed/
│
├── src/
│   └── data/
│       ├── __init__.py
│       └── fafb.py
│
├── scripts/
│   ├── inspect_fafb.py
│   ├── inspect_visual_system.py
│   ├── inspect_descending.py
│   ├── inspect_connections.py
│   ├── build_graph.py
│   ├── inspect_graph.py
│   ├── inspect_reachability.py
│   ├── inspect_encoder.py
│   ├── encode_image.py
│   └── train_classifier.py
│
├── tests/
│
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── results/
```

The project may gain additional packages under `src/` as the CRNN, visual encoder, and graph code are implemented. Keep modules named after real concepts such as `connectome`, `vision`, `models`, or `graph`, rather than generic abstraction layers.

---

## FAFB source data

Original FlyWire downloads live under:

```text
data/fafb/raw/
```

Expected files:

```text
connections_princeton.csv.gz
classification.csv.gz
consolidated_cell_types.csv.gz
neurons.csv.gz
visual_neuron_types.csv.gz
column_assignment.csv.gz
```

Keep them compressed. Pandas can read `.csv.gz` files directly.

Do not commit raw or processed datasets to Git.

### `connections_princeton.csv.gz`

Filtered synaptic connectivity.

Columns:

```text
pre_root_id
post_root_id
neuropil
syn_count
nt_type
```

Important details:

- roughly 5.34 million rows
- rows represent connected neuron pairs within a neuropil
- the same pre/post pair may occur in multiple neuropils
- the 5+ filter applies to total synapses across neuropils for a neuron pair
- therefore an individual row may have `syn_count < 5`
- do **not** apply another `syn_count >= 5` row filter

For CRNN v0, it is reasonable to aggregate by neuron pair:

```python
connections.groupby(
    ["pre_root_id", "post_root_id"]
)["syn_count"].sum()
```

Retain the original neuropil-resolved data for later analysis.

### `classification.csv.gz`

One row per FAFB neuron.

Columns:

```text
root_id
flow
super_class
class
sub_class
hemilineage
side
nerve
```

Use this for broad biological classification, including candidate descending/output populations.

### `consolidated_cell_types.csv.gz`

Columns:

```text
root_id
primary_type
additional_type(s)
```

Use for curated cell-type identity.

### `neurons.csv.gz`

Neurotransmitter predictions.

Includes:

```text
root_id
group
nt_type
nt_type_score
da_avg
ser_avg
gaba_avg
glut_avg
ach_avg
oct_avg
```

Do not initially assume neurotransmitter type maps trivially to excitatory/inhibitory sign. Receptor and circuit context matter.

### `visual_neuron_types.csv.gz`

Visual-neuron annotations.

Columns:

```text
root_id
type
family
subsystem
category
side
```

### `column_assignment.csv.gz`

Retinotopic visual-column assignments.

Columns:

```text
root_id
hemisphere
type
column_id
x
y
p
q
```

Observed dataset properties:

- 45,528 assigned neurons
- 31 columnar cell types
- 796 visual columns
- both hemispheres represented

Observed types:

```text
Mi1
L1
L2
L5
T4c
Tm1
Tm2
Mi9
T2a
Mi4
Tm9
C3
Tm3
T4d
T4b
T5b
Tm20
T3
T5c
T5a
Tm4
T2
T4a
C2
T5d
L3
T1
L4
Tm21
R8
R7
```

Do not treat all 31 as input neurons. Most are downstream visual-processing neurons.

R7 and R8 are present in the column map. R1-R6 are not present in this 31-type map, although FAFB contains reconstructed R1-R6 photoreceptors elsewhere in the annotations/connectivity.

---

## Data loading

`src/data/fafb.py` should be a thin source-data loader only.

Expected functions include:

```python
load_connections()
load_classification()
load_cell_types()
load_neurons()
load_visual_neurons()
load_visual_columns()
```

Keep graph construction, biological interpretation, feature extraction, and model code outside the loader.

Scripts should import from this module, e.g.:

```python
from data.fafb import load_connections
```

The repository should be installed editable:

```bash
pip install -e .
```

---

## CRNN model concept

The FAFB graph is sparse, directed, and recurrent.

Each FAFB neuron corresponds to one state/unit. Only FAFB-derived connections should exist in the biological graph.

A simple first recurrent model is:

```text
h(t+1) =
    (1 - leak) * h(t)
    + leak * activation(W_fafb * h(t) + external_input(t))
```

where:

- `h` is the neural state
- `W_fafb` is sparse
- topology comes from FAFB
- synapse count is only a proxy for initial connection strength
- the graph topology should initially stay fixed

A reasonable initial magnitude transform is:

```python
weight = log1p(syn_count)
```

Potentially normalize incoming weights to stabilize recurrence.

Do not model detailed spiking, ion channels, or membrane dynamics in v0.

---

## Visual input strategy

The image-to-fly mapping is a major modeling uncertainty.

Do not inject raw image values into every visual neuron type.

The preferred CRNN v0 approximation is:

```text
image/frame
    ↓
sample at FAFB retinotopic visual-column coordinates
    ↓
~796 visual samples
    ↓
simple temporal early-vision transform
    ↓
L1 / L2 / L3
    ↓
actual FAFB connectivity
```

A simple initial temporal encoding may be:

```text
ON(t)        = max(I(t) - I(t-1), 0)
OFF(t)       = max(I(t-1) - I(t), 0)
INTENSITY(t) = I(t)
```

with an approximate mapping such as:

```text
L1 <- ON / luminance-change signal
L2 <- OFF / luminance-change signal
L3 <- slower intensity / contrast signal
```

This is an engineering approximation, not a faithful model of fly phototransduction.

Keep the interface generic so it can later be replaced with a more biologically grounded R1-R6 photoreceptor model.

Suggested abstraction:

```python
class VisualEncoder:
    def encode(self, frame, previous_frame):
        ...
```

Possible later implementations:

```text
SimpleLaminaEncoder
PhotoreceptorEncoder
```

The CRNN should not depend on which encoder is used.

---

## Static-image datasets

Initial dataset may be Animals-10 or a simpler subset.

Suggested task progression:

```text
cat vs dog
cat / dog / horse / cow
Animals-10
```

Because fly visual processing is strongly temporal, static images may be converted into short active-vision sequences using small translations / zooms:

```text
frame 0: original
frame 1: small left shift
frame 2: small right shift
frame 3: small zoom
...
```

Keep such transformations reproducible.

---

## Output strategy

FAFB contains the brain, not the full ventral nerve cord.

The first biological output representation should therefore be a population of descending neurons rather than literal motor neurons.

Intended flow:

```text
FAFB state
    ↓
descending-neuron population
    ↓
small decoder
    ↓
animal class
```

Start with a small linear decoder:

```python
nn.Linear(num_descending_neurons, num_classes)
```

The first important classification experiment should freeze the CRNN and train only the decoder.

This asks whether animal category is linearly decodable from the representation produced by the fly-derived network.

---

## Work tracking

GitHub issues are the roadmap (what is next / what is done). This file
is the implementation contract (constraints, data, what not to do).

Do not duplicate the issue list here. Open issues at
https://github.com/tomluvoe/fafb-crnn/issues

Closed: Phase 0 (#5), Phase 1 (#6), Phase 2 (#7), Phase 3 (#8).
Next: Phase 4 CRNN (#9). Related: Animals-10 (#12), L1/L2 ON/OFF
encoding (#13; v0 default is spatial contrast).

## Remaining implementation constraints

These stay in-tree so agents do not need GitHub to implement correctly.

### Visual encoder

Sample images at FAFB retinotopic coordinates and inject approximate
ON / OFF / intensity into L1 / L2 / L3. Engineering stand-in, not
phototransduction. Keep a generic encoder interface so a later R1-R6
model can replace it.

### CRNN

Sparse recurrent propagation in PyTorch. Prefer sparse / edge-based
ops. Never construct a dense 139k × 139k matrix.

v0:

- fixed FAFB topology
- synapse-count-derived weights (`log1p` as proxy)
- simple rate dynamics
- no trainable internal weights
- 5 recurrent steps (max hops from visual inputs to reachable DNs)

### Classification

Train a small decoder on a frozen FAFB network first. Optionally later:
shared gains by cell type, limited trainable edge gains, strong
regularization, still-fixed topology.

### Controls

```text
raw sampled visual input -> linear classifier
random sparse network -> classifier
degree-preserving shuffled FAFB -> classifier
FAFB optic-lobe only -> classifier
FAFB optic-lobe + central brain -> classifier
frozen FAFB -> linear decoder
trainable FAFB CRNN
small conventional CNN baseline
```

A degree-preserving shuffled FAFB network is especially important.
Preserve as much as practical of in/out degree, weight distribution,
and possibly cell-type / transmitter stats, while randomizing exact
who-connects-to-whom. The point is whether exact FAFB wiring
contributes beyond generic sparse recurrence.

---

## Immediate priorities for coding agents

Follow open GitHub issues, starting with Phase 4 (#9). Do not redo
closed Phase 0–3 work unless those scripts are broken.

Do not guess biological labels. Query the actual CSV contents first.

---

## Interpretation discipline

Always distinguish:

### Directly from FAFB

Examples:

- neuron IDs
- cell annotations
- directed connectivity
- neuropil identity
- synapse counts
- retinotopic assignments
- neurotransmitter predictions

### Engineering assumptions

Examples:

- `log1p(syn_count)` as a neural-network weight
- rate-neuron dynamics
- leak constant
- activation function
- L1/L2/L3 image injection
- number of recurrent steps
- active-vision transforms
- classifier architecture

Results should be described in terms of the implemented connectome-constrained model, not as direct evidence that biological flies perform the same computation.
