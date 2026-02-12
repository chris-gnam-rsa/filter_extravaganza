from src import create_stars, Camera, Rotation
from src.dynamics import propagate
from src.earth import lla
from estimators import startracker
from estimators.errors import rotations2errors, plot_error_angles, filter_plot
from estimators.mekf import MEKF

import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

from rsalib.epoch import Epoch
from rsalib.satellites import TLESatelliteArray
from rsalib.units import Time

def main():
    # Simulation Settings:
    num_stars = 5000
    tsteps = 1000
    dt = 1
    animate = False
    time = Epoch(datetime.datetime(2026, 2, 12, 0, 0, 0))

    tle_catalog = "/Users/chrisgnam/source/repos/angl/angl/data/spacetrack_catalog.pkl"


    # Initial states:
    w0 = 0.5 * np.array([0.1, 0.1, 0.1]) * np.pi / 180  # rad/s
    q0 = np.array([0, 0, 0, 1])
    r0 = lla(0, 0, 400e3)  # m
    v0 = np.array([0, 0, 0])  #

    # Measurement noise:
    pixel_noise_std = 1.0  # pixels

    # Gyro noise parameters:
    sigma_u = 0.01 * np.pi / 180  # rad/s^0.5
    sigma_v = 0.001 * np.pi / 180  # rad/s / sqrt(s)

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
    X[0, 0:4] = q0
    X[0, 4:7] = w0

    # Filter Initialize:
    N = 3+3
    X_hat = np.zeros((tsteps, N))
    X_hat[0, 0:3] = np.zeros(3) # Initial attitude error guess
    X_hat[0, 3:6] = np.zeros(3) # Initial bias estimate

    angle_std = np.deg2rad(10)

    q_hat = np.zeros((tsteps, 4))
    q_hat[0] = Rotation.from_euler([angle_std * np.random.randn(), angle_std * np.random.randn(), angle_std * np.random.randn()], order="xyz").quaternion # Initial attitude guess
    # q_hat[0] = q0

    bias_std = 0.5
    P = np.diag([
        angle_std, angle_std, angle_std, 
        bias_std, bias_std, bias_std])**2 # Initial covariance guess
    sig3 = np.zeros((tsteps, N))
    sig3[0] = 3 * np.sqrt(np.diag(P))

    # Process noise covariance
    Q = np.zeros((6, 6))
    sigv2 = sigma_v**2
    sigu2 = sigma_u**2
    dt2 = dt**2
    dt3 = dt**3
    Q[0:3, 0:3] = (sigv2*dt + (1/3)*sigu2*dt3)*np.eye(3)
    Q[0:3, 3:6] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q[3:6, 0:3] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q[3:6, 3:6] = sigu2 * dt * np.eye(3)

    pixel_size = camera.sensor_size[0] / camera.resolution[0]  # mm/pixel
    star_vec_std = pixel_noise_std * pixel_size / camera.focal_length
    meas_std = np.array([star_vec_std]) # radians
    # meas_std[0] = 1e-1
    

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
        camera.position = r0

        # Simulate star measurements:
        star_pix, valid = camera.project_directions(stars)
        star_meas = star_pix + np.random.normal(0, pixel_noise_std, size=star_pix.shape)
        star_meas_vec = camera.pixel_to_rays_body(star_meas)
        star_true = stars[valid]

        [sat_r, sat_v, _] = satellites.propagate(time)
        sat_pix, valid = camera.project_points(sat_r)
        time = time + Time(dt, "second")


        # Simulate Gyro measurements:
        bias[i+1] = bias[i] + sigma_u * np.sqrt(dt) * np.random.randn()
        measured_rate[i+1] = w + bias[i+1] + sigma_v / np.sqrt(dt) * np.random.randn()

        # Run the MEKF:
        X_hat[i+1], P, q_hat[i+1] = MEKF(X_hat[i], P, q_hat[i], dt, meas_std, Q, star_meas_vec, star_true, measured_rate[i+1])
        sig3[i+1] = 3 * np.sqrt(np.diag(P))

        # Propagate the truth:
        X[i+1] = propagate(X[i], dt)

        # Optional animation:
        if animate:
            scatter.set_offsets(star_pix)
            scatter_star_meas.set_offsets(star_meas)
            scatter_sat.set_offsets(sat_pix)
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