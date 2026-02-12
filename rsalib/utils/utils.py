import numpy as np


def validate_array(vec: np.ndarray, size: int):
    if vec.shape != (size,):
        raise ValueError(f"Input must be a {size}D vector.")

    if not np.all(np.isfinite(vec)):
        raise ValueError("Input vector contains NaN or infinite values.")
