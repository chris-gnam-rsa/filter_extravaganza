from src import create_stars, Camera, Rotation
from src.dynamics import propagate
from src.earth import lla
from estimators import startracker
from estimators.errors import rotations2errors, plot_error_angles, filter_plot
from estimators import MEKF_attitude

import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

from rsalib.epoch import Epoch
from rsalib.satellites import TLESatelliteArray
from rsalib.units import Time

def main():
    # Simulation Settings:
    num_stars = 1000
    tsteps = 1000
    dt = 1
    animate = False
    time = Epoch(datetime.datetime(2026, 2, 12, 0, 0, 0))

    tle_catalog = "/Users/chrisgnam/source/repos/angl/angl/data/spacetrack_catalog.pkl"

    max_stars_used = 10

    # Initial states:
    cam_ar0 = np.array([0.1, 0.1, 0.1]) * np.pi / 180  # rad/s
    cam_quat0 = np.array([0, 0, 0, 1])
    cam_pos0 = lla(0, 0, 400e3)  # m
    cam_vek0 = np.array([0, 0, 0])  #

    # Measurement noise:
    pixel_noise_std = 1.0  # pixels

    # Gyroscope Noise Density (Angle Random Walk)
    # Spec: ~0.01 deg/s/sqrt(Hz)
    sig_arw = 0.01 * (np.pi / 180.0)  # ~1.74e-4 rad/s/sqrt(Hz)

    # Gyroscope Bias Diffusion (Rate Random Walk)
    # Spec: ~2.0 deg/hour/sqrt(Hz) ... hard to find exact, often tuned experimentally
    sig_rrw = 2.0e-5  # rad/s/sqrt(s)

    # Accelerometer Noise Density (Velocity Random Walk)
    # Spec: ~150 ug/sqrt(Hz) -> ~0.0015 m/s^2/sqrt(Hz)
    accel_noise_density = 150e-6 * 9.81  # ~0.00147 m/s^2/sqrt(Hz)

    # Accelerometer Bias Diffusion (Acceleration Random Walk)
    # A good starting point is usually 1/10th of the noise density magnitude or derived from instability
    accel_bias_random_walk = 1.0e-4  # m/s^2/sqrt(s)

    bias = np.zeros((tsteps, 3))
    measured_rate = np.zeros((tsteps, 3))
    bias[0] = np.zeros(3)


    # Create camera:
    camera = Camera(
        position=[0, 0, 0],
        orientation=Rotation.from_euler([0,0,0], order="xyz"),
        focal_length=50,
        sensor_size=[36, 24],
        resolution=[1920, 1080],
    )

    # Create the stars:
    satellites = TLESatelliteArray.from_file(Path(tle_catalog))
    [sat_r, sat_v, _] = satellites.propagate(time)
    
    stars = create_stars(num_stars)

    # Create the satellite manager:
    satellites = TLESatelliteArray.from_file(Path(tle_catalog))

    # Initialize:
    X = np.zeros((tsteps, 4+3))
    X[0, 0:4] = cam_quat0
    X[0, 4:7] = cam_ar0

    # Filter Initialize:
    N = 3+3
    X_hat = np.zeros((tsteps, N))
    X_hat[0, 0:3] = np.zeros(3) # Initial attitude error guess
    X_hat[0, 3:6] = np.zeros(3) # Initial bias estimate

    angle_std = np.deg2rad(5)

    q_hat = np.zeros((tsteps, 4))

    # Assume initialized with a static startracker solution, so small error
    tetra_error = np.deg2rad(0.5)
    q_hat[0] = Rotation.from_euler([tetra_error * np.random.randn(), tetra_error * np.random.randn(), tetra_error * np.random.randn()], order="xyz").quaternion

    bias_std = 0.05
    P = np.diag([
        angle_std, angle_std, angle_std, 
        bias_std, bias_std, bias_std])**2 # Initial covariance guess
    sig3 = np.zeros((tsteps, N))
    sig3[0] = 3 * np.sqrt(np.diag(P))

    # Process noise covariance
    Q = np.zeros((6, 6))
    sigu2 = sig_rrw**2
    sigv2 = sig_arw**2
    dt2 = dt**2
    dt3 = dt**3
    Q[0:3, 0:3] = (sigv2*dt + (1/3)*sigu2*dt3)*np.eye(3)
    Q[0:3, 3:6] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q[3:6, 0:3] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q[3:6, 3:6] = sigu2 * dt * np.eye(3)

    meas_std = np.array([pixel_noise_std])
    
    # Camera model parameters:
    ax = camera.focal_length * camera.resolution[0] / camera.sensor_size[0]
    ay = camera.focal_length * camera.resolution[1] / camera.sensor_size[1]
    u0 = camera.resolution[0] / 2
    v0 = camera.resolution[1] / 2

    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter([], [], s=6, color="white", marker="o")
        scatter_star_meas = ax.scatter([], [], s=10, color="red", alpha=0.5, marker="x")
        scatter_sat = ax.scatter([], [], s=3, color="cyan", marker="o")
        ax.set_xlim(0, camera.resolution[0])
        ax.set_ylim(0, camera.resolution[1])
        ax.set_facecolor("black")
        ax.set_title(f"Projection of {num_stars} Stars")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.set_aspect("equal", adjustable="box")

    for i in range(tsteps-1):
        # Update the camera:
        q = X[i, 0:4]
        w = X[i, 4:7]
        camera.orientation = Rotation.from_quaternion(q)
        camera.position = cam_pos0

        # Simulate star measurements:
        star_pix, valid = camera.project_directions(stars)
        star_meas = star_pix + np.random.normal(0, pixel_noise_std, size=star_pix.shape)

        # Obtain corresponding true stars
        star_true = stars[valid] 

        # Limit to number of stars used by filter:
        if star_meas.shape[0] > max_stars_used:
            curr_star_meas_pix = star_meas[0:max_stars_used, :]
            curr_star_true = star_true[0:max_stars_used, :]
        else:
            curr_star_meas_pix = star_meas
            curr_star_true = star_true

        # [sat_r, sat_v, _] = satellites.propagate(time)
        # sat_pix, valid = camera.project_points(sat_r)
        # time = time + Time(dt, "second")


        # Simulate Gyro measurements:
        bias[i+1] = bias[i] + sig_rrw * np.sqrt(dt) * np.random.randn()
        measured_rate[i+1] = w + bias[i+1] + sig_arw / np.sqrt(dt) * np.random.randn()

        # Run the MEKF:
        X_hat[i+1], P, q_hat[i+1] = MEKF_attitude(
            X_hat[i], P, q_hat[i], dt, meas_std, Q, 
            curr_star_meas_pix,
            curr_star_true, 
            measured_rate[i+1], 
            ax, ay, u0, v0
        )
        sig3[i+1] = 3 * np.sqrt(np.diag(P))

        # Propagate the truth:
        X[i+1] = propagate(X[i], dt)

        # Optional animation:
        if animate:
            scatter.set_offsets(star_pix)
            scatter_star_meas.set_offsets(star_meas)
            # scatter_sat.set_offsets(sat_pix)
            fig.canvas.draw_idle()
            plt.pause(0.01)

    if animate:
        plt.ioff()
        plt.show()

    # Plot the results:
    t_array = np.arange(tsteps).flatten()*dt
    
    q_true = X[:, 0:4]
    q_hat_sig3 = sig3[:, 0:3]

    plot_error_angles(t_array, q_hat, q_true, q_hat_sig3)

    plt.figure()
    plt.subplot(3, 1, 1)
    filter_plot(t_array, X_hat[:, 3], sig3[:, 3], "Bias X (rad/s)")
    plt.title("Gyro Bias Estimation Errors")

    plt.subplot(3, 1, 2)
    filter_plot(t_array, X_hat[:, 4], sig3[:, 4], "Bias Y (rad/s)")

    plt.subplot(3, 1, 3)
    filter_plot(t_array, X_hat[:, 5], sig3[:, 5], "Bias Z (rad/s)")
    plt.show()


if __name__ == "__main__":
    main()