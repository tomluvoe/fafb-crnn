from pathlib import Path

import pandas as pd


# Repository root: fafb-crnn/
REPO_ROOT = Path(__file__).resolve().parents[2]
FAFB_RAW_DIR = REPO_ROOT / "data" / "fafb" / "raw"


def load_connections() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "connections_princeton.csv.gz"
    )


def load_classification() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "classification.csv.gz"
    )


def load_cell_types() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "consolidated_cell_types.csv.gz"
    )


def load_neurons() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "neurons.csv.gz"
    )


def load_visual_neurons() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "visual_neuron_types.csv.gz"
    )


def load_visual_columns() -> pd.DataFrame:
    return pd.read_csv(
        FAFB_RAW_DIR / "column_assignment.csv.gz"
    )
