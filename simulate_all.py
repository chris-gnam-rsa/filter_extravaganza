from src import create_stars, Camera, Rotation
from src.dynamics import propagate
from src.earth import lla
from estimators.errors import plot_error_angles, filter_plot
from estimators import MEKF_full

import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

from rsalib.epoch import Epoch
from rsalib.satellites import TLESatelliteArray
from rsalib.units import Time

def main():
    ###########################
    ### Simulation Settings ###
    ###########################
    dt = 0.1                 # simulation step
    camera_dt = 10.0         # camera period in seconds
    duration = 10*60         # total simulation duration in seconds

    num_stars = 5000
    max_stars_used = 10

    time = Epoch(datetime.datetime(2026, 2, 12, 0, 0, 0))
    tle_catalog = "/Users/chrisgnam/source/repos/angl/angl/data/spacetrack_catalog.pkl"


    ############################
    ### Camera Configuration ###
    ############################   
    focal_length = 50
    sensor_size = [36, 24]
    resolution = [1920, 1080]


    ######################
    ### Initial states ###
    ######################
    cam_ar0 = np.array([0.1, 0.1, 0.1]) * np.pi / 180  # rad/s
    cam_quat0 = np.array([0, 0, 0, 1])
    cam_pos0 = lla(0, 0, 400e3)  # m
    cam_vel0 = np.array([0, 0, 0]) 


    ####################
    ### Sensor Noise ###
    ####################
    pixel_noise_std = 1.0  # Camera centroiding noise in pixels

    # Gyroscope Noise Density (Angle Random Walk)
    # Spec: ~0.01 deg/s/sqrt(Hz)
    sig_arw = 0.01 * (np.pi / 180.0)  # ~1.74e-4 rad/s/sqrt(Hz)

    # Gyroscope Bias Diffusion (Rate Random Walk)
    # Spec: ~2.0 deg/hour/sqrt(Hz) ... hard to find exact, often tuned experimentally
    sig_rrw = 2.0e-5  # rad/s/sqrt(s)




    ##################
    ### Initialize ###
    ##################
    camera = Camera(
        position=[0, 0, 0],
        orientation=Rotation.from_euler([0,0,0], order="xyz"),
        focal_length=focal_length,
        sensor_size=sensor_size,
        resolution=resolution,
    )

    # Camera model parameters used in filter:
    ax = camera.focal_length * camera.resolution[0] / camera.sensor_size[0]
    ay = camera.focal_length * camera.resolution[1] / camera.sensor_size[1]
    u0 = camera.resolution[0] / 2
    v0 = camera.resolution[1] / 2

    camera_step = int(round(camera_dt / dt))
    if camera_step < 1:
        raise ValueError("camera_dt must be >= dt")
    


    # Create the stars:    
    stars = create_stars(num_stars)

    # Load the satellite catalog:
    satellites = TLESatelliteArray.from_file(Path(tle_catalog))


    ####################
    ### Pre-allocate ###
    ####################
    tsteps = int(duration / dt) + 1

    # Initialize gyro bias and measurement arrays:
    measured_rate = np.zeros((tsteps, 3))
    gyro_bias = np.zeros((tsteps, 3))
    gyro_bias[0] = np.zeros(3)

    # Initialize true state array:
    X = np.zeros((tsteps, 4+3))
    X[0, 0:4] = cam_quat0
    X[0, 4:7] = cam_ar0

    # Initialize filter estimate array:
    X_hat = np.zeros((tsteps, 3+3))
    sig3 = np.zeros((tsteps, 3+3))

    X_hat[0, 0:3] = np.zeros(3) # Initial attitude error guess
    X_hat[0, 3:6] = np.zeros(3) # Initial bias estimate

    q_hat = np.zeros((tsteps, 4))

    tetra_error = np.deg2rad(0.5)
    q_hat[0] = Rotation.from_euler(tetra_error * np.random.randn(3,1), order="xyz").quaternion


    # Initial estimation covariance:
    angle_std = np.deg2rad(5)
    gyro_bias_std = 0.05
    P = np.diag([
        angle_std, angle_std, angle_std, 
        gyro_bias_std, gyro_bias_std, gyro_bias_std])**2 # Initial covariance guess
    
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


    #####################
    ### Simulate Loop ###
    #####################
    for i in range(tsteps-1):
        # Update the camera:
        q = X[i, 0:4]
        w = X[i, 4:7]
        camera.orientation = Rotation.from_quaternion(q)
        camera.position = cam_pos0

        # Simulate camera measurements (once every camera_step steps):
        if (i % camera_step) != 0:
            curr_star_meas_pix = np.array([])
            curr_star_true = np.array([])

            sat_meas = np.array([])
            sat_r_true = np.array([])

        else:
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

            # Simulate satellite measurements:
            [sat_r, sat_v, _] = satellites.propagate(time)
            sat_pix, valid = camera.project_points(sat_r)
            sat_meas = sat_pix + np.random.normal(0, pixel_noise_std, size=sat_pix.shape)
            sat_r_true = sat_r[valid]

            # TODO apply noise to the satellite position (catalog error):



        time = time + Time(dt, "second")

        # Simulate Gyro measurements (every step):
        gyro_bias[i+1] = gyro_bias[i] + sig_rrw * np.sqrt(dt) * np.random.randn()
        measured_rate[i+1] = w + gyro_bias[i+1] + sig_arw / np.sqrt(dt) * np.random.randn()

        # Run the MEKF:
        X_hat[i+1], P, q_hat[i+1] = MEKF_full(
            X_hat[i], P, q_hat[i], dt, meas_std, Q, 
            curr_star_meas_pix,
            curr_star_true,
            sat_meas,
            sat_r_true,
            measured_rate[i+1], 
            ax, ay, u0, v0
        )
        sig3[i+1] = 3 * np.sqrt(np.diag(P))

        # Propagate the truth:
        X[i+1] = propagate(X[i], dt)


    # Plot the results:
    t_array = np.arange(0, duration + dt, dt).flatten()
    
    q_true = X[:, 0:4]
    q_hat_sig3 = sig3[:, 0:3]
    plot_error_angles(t_array, q_hat, q_true, q_hat_sig3)

    bias_error = np.rad2deg(X_hat[:, 3:6] - gyro_bias)
    bias_sig3 = np.rad2deg(sig3[:, 3:6])

    plt.figure()
    plt.subplot(3, 1, 1)
    # Plot ERROR vs BOUNDS
    filter_plot(t_array, bias_error[:, 0], bias_sig3[:, 0], "Bias X Error (deg/s)")
    plt.title("Gyro Bias Estimation Errors")

    plt.subplot(3, 1, 2)
    filter_plot(t_array, bias_error[:, 1], bias_sig3[:, 1], "Bias Y Error (deg/s)")

    plt.subplot(3, 1, 3)
    filter_plot(t_array, bias_error[:, 2], bias_sig3[:, 2], "Bias Z Error (deg/s)")
    plt.show()


if __name__ == "__main__":
    main()