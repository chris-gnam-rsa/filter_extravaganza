import numpy as np


def ric_basis(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Build the RIC (radial, in-track, cross-track) basis in ECI coordinates.

    Args:
        r_eci (np.ndarray): Position array with last dimension size 3.
        v_eci (np.ndarray): Velocity array with last dimension size 3.

    Returns:
        np.ndarray: Basis matrix with shape (..., 3, 3). Columns are [R, I, C].
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)

    if r.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("Input position/velocity must have last dimension size 3.")

    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    if np.any(r_norm == 0):
        raise ValueError("Position vector magnitude must be non-zero.")

    r_hat = r / r_norm
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=-1, keepdims=True)
    if np.any(h_norm == 0):
        raise ValueError("Angular momentum magnitude must be non-zero.")

    c_hat = h / h_norm
    i_hat = np.cross(c_hat, r_hat)

    return np.stack((r_hat, i_hat, c_hat), axis=-1)


def apply_ric_offsets(
    r_eci: np.ndarray, v_eci: np.ndarray, ric_offsets: np.ndarray
) -> np.ndarray:
    """
    Apply RIC offsets to position vectors in a physically consistent way.

    Args:
        r_eci (np.ndarray): Position array with last dimension size 3.
        v_eci (np.ndarray): Velocity array with last dimension size 3.
        ric_offsets (np.ndarray): Offsets in RIC [radial, in-track, cross-track].

    Returns:
        np.ndarray: Offset position array in ECI with the same shape as r_eci.
    """
    r = np.asarray(r_eci, dtype=float)
    offsets = np.asarray(ric_offsets, dtype=float)

    if offsets.shape[-1] != 3:
        raise ValueError("RIC offsets must have last dimension size 3.")

    basis = ric_basis(r_eci=r, v_eci=v_eci)
    offset_eci = np.einsum("...ij,...j->...i", basis, offsets)
    return r + offset_eci
