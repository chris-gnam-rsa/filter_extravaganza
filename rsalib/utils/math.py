import numpy as np


def skew_symmetric(v: np.array) -> np.array:
    """
    Constructs a skew-symmetric matrix from a 3D vector.

    Args:
        v (numpy.array): 3-element vector (shape: (3,))

    Returns:
        numpy.array: 3x3 skew-symmetric matrix
    """
    assert v.shape == (3,), "Input must be a 3-element vector"

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
