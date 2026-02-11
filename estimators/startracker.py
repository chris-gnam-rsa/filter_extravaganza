from src.math import skew

import numpy as np

def compute_attitude_covariance(b_body: list[np.ndarray], sigma_pix: np.array, focal_length_pix: float) -> np.ndarray:

    var_u = (sigma_pix[0] / focal_length_pix) ** 2
    var_v = (sigma_pix[1] / focal_length_pix) ** 2
    
    info = np.zeros((3, 3))
    
    for b in b_body:
        # From linear approximation:
        H = skew(b)
        
        R = var_u * np.outer(H[:, 0], H[:, 0]) + var_v * np.outer(H[:, 1], H[:, 1])
        R += 1e-12 * np.outer(b, b)
        
        R_inv = np.linalg.inv(R)
        info += H.T @ R_inv @ H
    
    P = np.linalg.inv(info)
    
    return P

def startracker(detected_stars, vec_meas, meas_uncertainty=None, focal_length_pix=10):
    # Assumes input vectors are normalized
    assert detected_stars.shape == vec_meas.shape

    # Compute optimal rotation using SVD:
    H = vec_meas.T @ detected_stars
    U, S, Vt = np.linalg.svd(H)
    R_opt = U @ Vt
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = U @ Vt

    # Assume a very small uncertainty if none is provided:
    if meas_uncertainty is None:
        R_cov_opt = np.diag([1e-6, 1e-6, 1e-6])

    else:
        # Propagate uncertainties into the rotation estimate:
        R_cov_opt = compute_attitude_covariance(detected_stars, meas_uncertainty, focal_length_pix)

    return R_opt, R_cov_opt