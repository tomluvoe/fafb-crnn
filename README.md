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

Raw FlyWire tables go in `data/fafb/raw/` (not in git). Every inspect
and preprocess step is a script; re-run them at any time. `build_graph.py`
is the only writer.

```bash
uv run python scripts/inspect_fafb.py
uv run python scripts/inspect_visual_system.py
uv run python scripts/inspect_descending.py
uv run python scripts/inspect_connections.py
uv run python scripts/build_graph.py
uv run python scripts/inspect_graph.py
```

`uv run` creates `.venv` and installs this package if needed. System
`python3` does not see `src/`.

| script | what it does |
|---|---|
| `inspect_fafb.py` | first pass over every raw table |
| `inspect_visual_system.py` | photoreceptors, L1/L2/L3, retinotopy |
| `inspect_descending.py` | v0 readout: `super_class == descending` |
| `inspect_connections.py` | pair aggregation, isolated cells, `syn_count` |
| `build_graph.py` | writes `data/fafb/processed/` (deterministic) |
| `inspect_graph.py` | checks the processed artifacts without re-parsing CSV |

## Lint and tests

PRs run [Ruff](https://docs.astral.sh/ruff/) and pytest in GitHub
Actions. Locally:

```bash
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest
```

Inspect scripts are not unit-tested in CI because the `.csv.gz` files
are local-only. Graph index tests use tiny synthetic graphs.

## Status

Phase 1: deterministic `root_id → index` mapping and saved sparse graph.

## Interpretation

Results describe the implemented connectome-constrained model, not
direct evidence that biological flies perform the same computation.
Engineering choices (rate dynamics, L1/L2/L3 image injection, leak,
decoder) are approximations and should be labeled as such.
