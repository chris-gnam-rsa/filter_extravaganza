import numpy as np

from src.math import skew
from src import Rotation
from src.dynamics import propagate_quaternion


def Xi(q):
    q1, q2, q3, q4 = q
    return np.array([
        [ q4, -q3,  q2],
        [ q3,  q4, -q1],
        [-q2,  q1,  q4],
        [-q1, -q2, -q3]
    ])


def MEKF(X_hat, P, q_hat, dt, meas_std, Q, star_meas_vec, star_true, measured_rate):
    sigma_star = meas_std[0]

    # Get rotation matrix from quaternion
    A_body_to_inertial = Rotation.from_quaternion(q_hat).matrix
    A_inertial_to_body = A_body_to_inertial.T
    
    # Predicted star directions in body frame
    predict = (A_inertial_to_body @ star_true.T).T
    
    # Innovation:
    innovation = star_meas_vec.flatten() - predict.flatten()
    
    # Construct measurement Jacobian and covariance
    n_stars = star_meas_vec.shape[0]
    H = np.zeros((3 * n_stars, 6))
    R_diag = np.zeros(3 * n_stars)
    
    for i in range(n_stars):
        H[3*i:3*(i+1), 0:3] = skew(predict[i, :])
        H[3*i:3*(i+1), 3:6] = np.zeros((3, 3))
        R_diag[3*i:3*(i+1)] = sigma_star**2
    
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
    q_hat = q_hat + 0.5 * Xi(q_hat) @ da
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