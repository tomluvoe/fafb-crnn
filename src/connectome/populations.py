"""v0 FAFB populations used as CRNN input and readout.

These filters are read off the FlyWire tables. They are not a claim that
the biological categories are complete.
"""

import numpy as np
import pandas as pd

V0_VISUAL_INPUT_TYPES = ("L1", "L2", "L3")
V0_DESCENDING_SUPER_CLASS = "descending"
OPTIC_SUPER_CLASS = "optic"
VISUAL_PROJECTION_SUPER_CLASS = "visual_projection"


def descending_neurons(classification: pd.DataFrame) -> pd.DataFrame:
    """Neurons with super_class == descending."""
    return classification.loc[
        classification["super_class"].eq(V0_DESCENDING_SUPER_CLASS)
    ].copy()


def visual_input_columns(visual_columns: pd.DataFrame) -> pd.DataFrame:
    """Column-assigned L1 / L2 / L3 neurons used as v0 visual inputs."""
    return visual_columns.loc[visual_columns["type"].isin(V0_VISUAL_INPUT_TYPES)].copy()


def super_class_mask(
    classification: pd.DataFrame,
    neuron_ids: np.ndarray,
    super_classes: tuple[str, ...],
) -> np.ndarray:
    """Boolean mask over graph indices for the given super_class labels."""
    wanted = set(
        classification.loc[
            classification["super_class"].isin(super_classes), "root_id"
        ].tolist()
    )
    return np.isin(neuron_ids, list(wanted))
