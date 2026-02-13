import numpy as np
from matplotlib import pyplot as plt

from src.rotation import Rotation

def rotations2errors(R_true: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    R_err = R_est @ R_true.T
    angle = np.arccos((np.trace(R_err) - 1) / 2)

    if np.isclose(angle, 0):
        return np.zeros(3)
    elif np.isclose(angle, np.pi):
        # Handle the singularity at 180 degrees
        R_err_plus_I = R_err + np.eye(3)
        axis = np.sqrt(np.diag(R_err_plus_I) / 2)
        axis /= np.linalg.norm(axis)
        return axis * angle
    else:
        rx = R_err[2,1] - R_err[1,2]
        ry = R_err[0,2] - R_err[2,0]
        rz = R_err[1,0] - R_err[0,1]
        axis = np.array([rx, ry, rz])
        axis /= (2 * np.sin(angle))
        return axis * angle
    

def filter_plot(time_array, error_array, sigma_array, ylabel, scale=None):
    plt.plot(time_array, error_array)
    plt.fill_between(time_array,
                     -1 * sigma_array,
                     1 * sigma_array,
                     color='green', alpha=0.25, label='1-sigma bounds')
    plt.fill_between(time_array,
                     -2 * sigma_array,
                     2 * sigma_array,
                     color='yellow', alpha=0.2, label='2-sigma bounds')
    plt.fill_between(time_array,
                     -3 * sigma_array,
                     3 * sigma_array,
                     color='red', alpha=0.15, label='3-sigma bounds')
    plt.ylabel(ylabel)
    plt.grid()

    if scale is not None:
        plt.ylim(-np.median(3*sigma_array)*scale, np.median(3*sigma_array)*scale)


def plot_error_angles(time_array, q_hat, q_true, q_hat_sig3, scale=None):
    error_angles = np.zeros((q_hat.shape[0], 3))
    for i in range(q_hat.shape[0]):
        R_true = Rotation.from_quaternion(q_true[i]).matrix
        R_est = Rotation.from_quaternion(q_hat[i]).matrix
        error_angles[i] = rotations2errors(R_true, R_est)

    error_angles_deg = np.rad2deg(error_angles)
    q_hat_sig3_deg = np.rad2deg(q_hat_sig3)

    plt.figure()
    plt.subplot(3, 1, 1)
    filter_plot(time_array, error_angles_deg[:,0], q_hat_sig3_deg[:,0], "Roll Error (deg)", scale=scale)
    plt.title("Star Tracker Attitude Estimation Errors")

    plt.subplot(3, 1, 2)
    filter_plot(time_array, error_angles_deg[:,1], q_hat_sig3_deg[:,1], "Pitch Error (deg)", scale=scale)
    plt.ylabel("Angle Error (deg)")

    plt.subplot(3, 1, 3)
    filter_plot(time_array, error_angles_deg[:,2], q_hat_sig3_deg[:,2], "Yaw Error (deg)", scale=scale)