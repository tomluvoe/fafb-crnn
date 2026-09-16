# fafb-crnn

Connectome-constrained recurrent neural network (CRNN) built from the
FlyWire **FAFB** dataset: the Female Adult Fly Brain connectome of
*Drosophila melanogaster*.

The experiment asks whether the fly's visual wiring produces
representations from which high-level visual categories can be decoded.
It does **not** ask whether a fly can recognize cats.

## Pipeline

```text
image / short frame sequence
    → fly-inspired visual encoder
    → identified FAFB visual input neurons
    → FAFB recurrent network (optic lobe + central brain)
    → descending neurons
    → small linear decoder
    → animal class
```

v0 is deliberately simple:

- one FAFB neuron → one recurrent unit
- only FAFB-derived directed edges
- synapse count as a proxy for initial weight (`log1p`), not biology
- images sampled at retinotopic visual-column coordinates
- approximate ON / OFF / intensity signals injected into **L1 / L2 / L3**
- frozen connectome; train only a linear readout from descending neurons

Later work can replace the encoder, add limited trainable gains, and run
controls (shuffled wiring, optic-lobe-only, CNN baseline).

## Data

Raw FlyWire tables live in `data/fafb/raw/` and are not committed:

- `connections_princeton.csv.gz` — synaptic connectivity
- `classification.csv.gz` — neuron taxonomy
- `consolidated_cell_types.csv.gz` — curated cell types
- `neurons.csv.gz` — neurotransmitter predictions
- `visual_neuron_types.csv.gz` — visual-neuron annotations
- `column_assignment.csv.gz` — retinotopic visual columns

## Scripts

```bash
uv run python scripts/inspect_fafb.py
uv run python scripts/inspect_visual_system.py
```

`uv run` creates `.venv` and installs this package if needed. System
`python3` does not see `src/`.

`inspect_fafb.py` is a first pass over every table.
`inspect_visual_system.py` looks at photoreceptors, lamina neurons,
retinotopy, and which cells are usable as CRNN visual inputs.

## Lint

PRs run [Ruff](https://docs.astral.sh/ruff/) in GitHub Actions. Locally:

```bash
uv run ruff check src scripts
uv run ruff format --check src scripts
```

This does not load FAFB data. Inspect scripts are not unit-tested in CI
because the `.csv.gz` files are local-only. Graph index tests can be
added later against tiny synthetic graphs.

## Status

Phase 0: inspect the connectome and visual annotations before building
the graph or the model.

## Interpretation

Results describe the implemented connectome-constrained model, not
direct evidence that biological flies perform the same computation.
Engineering choices (rate dynamics, L1/L2/L3 image injection, leak,
decoder) are approximations and should be labeled as such.
