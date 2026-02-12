import numpy as np

from src.math import skew, xi
from src import Rotation
from src.dynamics import propagate_quaternion


def MEKF_full(X_hat, P, q_hat, dt, meas_std, Q, star_meas_pix, star_true, sat_meas_pix, sat_true, measured_rate, ax, ay, u0, v0):
    # Extract measurement uncertainties:
    sigma_pixel = meas_std[0]  # Standard deviation in PIXELS

    n_stars = star_meas_pix.shape[0]
    if n_stars != 0:
        # Recover attitude:
        A_body_to_inertial = Rotation.from_quaternion(q_hat).matrix
        A_inertial_to_body = A_body_to_inertial.T
        
        # Predict:
        b_pred = (A_inertial_to_body @ star_true.T).T
    
    # Initialization for 2D measurements (2 rows per star)
    H = np.zeros((2 * n_stars, 6))
    innovation = np.zeros(2 * n_stars)
    R_diag = np.zeros(2 * n_stars)
    
    for i in range(n_stars):
        # Measurement prediction for star i:
        x, y, z = b_pred[i]
        u_pred = ax * (x / z) + u0
        v_pred = v0 - ay * (y / z)
        
        # Innovation (Measurement - Prediction)
        innovation[2*i]     = star_meas_pix[i, 0] - u_pred
        innovation[2*i + 1] = star_meas_pix[i, 1] - v_pred

        # Projective Jacobian
        term1 = np.array([
            [ax / z, 0,      -ax * x / z**2], 
            [0,      -ay / z, ay * y / z**2]
        ])

        # Kinematic Jacobian
        term2 = skew(b_pred[i])
        
        H_att = term1 @ term2

        # Construct matrix:
        H[2*i : 2*i+2, 0:3] = H_att
        H[2*i : 2*i+2, 3:6] = np.zeros((2, 3)) # Bias partials are 0
        
        # Measurement Noise
        R_diag[2*i]     = sigma_pixel**2
        R_diag[2*i + 1] = sigma_pixel**2
        
    R = np.diag(R_diag)

    # Kalman gain
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # State correction
    delta_x = K @ innovation
    da = delta_x[0:3]  # Attitude error correction
    db = delta_x[3:6]  # Bias correction
    
    # Quaternion update (Crassidis eq. 7.34)
    # q+ = q- + 0.5 * Xi(q-) * da
    q_hat = q_hat + 0.5 * xi(q_hat) @ da
    q_hat = q_hat / np.linalg.norm(q_hat)

    # Bias update
    bias_hat = X_hat[3:6] + db

    # Covariance update (Joseph form for numerical stability)
    I_KH = np.eye(6) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    
    # Corrected angular rate (measured - bias)
    omega = measured_rate - bias_hat
    
    # Propagate quaternion
    q_hat = propagate_quaternion(q_hat, omega, dt)
    q_hat = q_hat / np.linalg.norm(q_hat)
    
    # Discrete-time Process noise mapping
    G = np.zeros((6, 6))
    G[0:3, 0:3] = -np.eye(3)  # Attitude noise
    G[3:6, 3:6] = np.eye(3)   # Bias noise
    
    # State transition matrix
    w = np.linalg.norm(omega)
    wx = skew(omega)
    wx2 = wx @ wx
    Phi_att_11 = np.eye(3) - (wx * np.sin(w*dt)/w) + (wx2 * ((1 - np.cos(w * dt)))/(w**2))
    Phi_att_12 = (wx*((1 - np.cos(w*dt))/(w**2))) - (np.eye(3)*dt) - (wx2*(w*dt - np.sin(w*dt))/(w**3))
    Phi_att_21 = np.zeros((3, 3))
    Phi_att_22 = np.eye(3)

    Phi_att = np.block([
        [Phi_att_11, Phi_att_12],
        [Phi_att_21, Phi_att_22]
    ])

    Phi = Phi_att

    # Discrete-time covariance propagation
    P = Phi @ P @ Phi.T + G @ Q @ G.T
    
    # Reset
    X_hat_out = np.zeros(6)
    X_hat_out[3:6] = bias_hat
    
    return X_hat_out, P, q_hat