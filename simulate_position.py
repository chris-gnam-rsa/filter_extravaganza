from src import create_stars, Camera, Rotation
from src.dynamics import propagate
from src.earth import lla
from src.orbit_utils import apply_ric_offsets
from estimators.errors import plot_error_angles, filter_plot
from estimators import MEKF_position

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
    dt = 0.1                # simulation step
    camera_dt = 1.0         # camera period in seconds
    duration = 60*60        # total simulation duration in seconds

    num_stars = 5000
    max_stars_used = 10

    time = Epoch(datetime.datetime(2026, 2, 12, 0, 0, 0))
    tle_catalog = "/Users/chrisgnam/source/repos/angl/angl/data/spacetrack_catalog.pkl"

    animate = False

    fraction_images_unavailable = 0.05 # Fraction of images where star measurements are unavailable

    # sat_intrack_std = 4000*1000/3 # 4000km 3-sigma (converted to standard deviation)
    # sat_crosstrack_std = 5*1000/3 # 5km 3-sigma (converted to standard deviation)
    # sat_radial_std = 40*1000/3    # 40km 3-sigma (converted to standard deviation)

    sat_intrack_std = 40*1000/3 # 4000km 3-sigma (converted to standard deviation)
    sat_crosstrack_std = 1*1000/3 # 5km 3-sigma (converted to standard deviation)
    sat_radial_std = 10*1000/3    # 40km 3-sigma (converted to standard deviation)

    ############################
    ### Camera Configuration ###
    ############################   
    focal_length = 50
    sensor_size = [36, 24]
    resolution = [1920, 1080]


    ######################
    ### Initial states ###
    ######################
    cam_pos0 = lla(0, 0, 400e3)  # m
    cam_vel0 = np.array([0, 0, 0]) 

    # Generate an upwards pointing quaternion:
    cam_pos_unit = cam_pos0 / np.linalg.norm(cam_pos0)
    true_rotation = Rotation.align_vectors(np.array([[0, 0, 1]]), np.array([cam_pos_unit]))
    cam_quat0 = true_rotation.quaternion
    cam_ar0 = np.deg2rad(np.array([0.01, 0.02, 0.03]))



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

    # Camera available history:
    camera_available_history = np.zeros(tsteps, dtype=bool)

    # Initialize gyro bias and measurement arrays:
    measured_rate = np.zeros((tsteps, 3))
    gyro_bias = np.zeros((tsteps, 3))
    gyro_bias[0] = np.zeros(3)

    # Initialize true state array:
    X = np.zeros((tsteps, 4+3))
    X[0, 0:4] = cam_quat0
    X[0, 4:7] = cam_ar0

    # Initialize filter estimate array:
    N = 3 + 3 + 3 # Position, Attitude Error, Gyro Bias
    X_hat = np.zeros((tsteps, N)) 
    sig3 = np.zeros((tsteps, N))

    X_hat[0, 0:3] = cam_pos0 + np.random.randn(3) * 10000 # Initial position estimate
    X_hat[0, 3:6] = np.zeros(3) # Initial attitude error
    X_hat[0, 6:9] = np.zeros(3) # Initial bias estimate

    q_hat = np.zeros((tsteps, 4))

    tetra_error = np.deg2rad(0.5)
    error_rotation = Rotation.from_euler(tetra_error * np.random.randn(3), order="xyz")
    q_hat[0] = Rotation.from_matrix(error_rotation.matrix @ true_rotation.matrix).quaternion


    # Initial estimation covariance:
    pos_std = 100000
    angle_std = np.deg2rad(5)
    gyro_bias_std = 0.05
    P = np.diag([
        pos_std, pos_std, pos_std,
        angle_std, angle_std, angle_std, 
        gyro_bias_std, gyro_bias_std, gyro_bias_std])**2 # Initial covariance guess
    
    sig3[0] = 3 * np.sqrt(np.diag(P))

    # Process noise covariance
    Q_pos = np.diag([1e-9, 1e-9, 1e-9])

    Q_att = np.zeros((6, 6))
    sigu2 = sig_rrw**2
    sigv2 = sig_arw**2
    dt2 = dt**2
    dt3 = dt**3
    Q_att[0:3, 0:3] = (sigv2*dt + (1/3)*sigu2*dt3)*np.eye(3)
    Q_att[0:3, 3:6] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q_att[3:6, 0:3] = 0.5 * sigu2 * dt2 * np.eye(3)
    Q_att[3:6, 3:6] = sigu2 * dt * np.eye(3)

    Q = np.block([
        [Q_pos, np.zeros((3, 6))],
        [np.zeros((6, 3)), Q_att]
    ])

    star_pixel_noise_std = pixel_noise_std
    sat_pixel_noise_std = pixel_noise_std + 15
    meas_std = np.array([star_pixel_noise_std, sat_pixel_noise_std])

    sat_offsets_initialized = False


    #####################
    ### Simulate Loop ###
    #####################
    if animate:
        plt.ion()
        fig, ax_anim = plt.subplots()
        fig.patch.set_facecolor("black")
        ax_anim.set_facecolor("black")
        star_plt = ax_anim.scatter([], [], s=10, c="white")
        sat_plt = ax_anim.scatter([], [], s=5, c="cyan")
        sat_predict_plt = ax_anim.scatter(
            [],
            [],
            s=200,
            facecolors="none",
            edgecolors="magenta",
            linewidths=1.0,
            marker="o",
        )
        ax_anim.set_xlim(0, camera.resolution[0])
        ax_anim.set_ylim(0, camera.resolution[1])
        ax_anim.set_aspect("equal", adjustable="box")
        ax_anim.invert_yaxis()
        ax_anim.set_xticks([])
        ax_anim.set_yticks([])
        ax_anim.set_frame_on(False)

    for i in range(tsteps-1):
        # Update the camera:
        q = X[i, 0:4]
        w = X[i, 4:7]
        camera.orientation = Rotation.from_quaternion(q)
        camera.position = cam_pos0

        # Simulate camera measurements (once every camera_step steps):
        camera_available = (i % camera_step) == 0 and (np.random.rand() > fraction_images_unavailable)
        camera_available_history[i] = camera_available
        if camera_available:
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

            if not sat_offsets_initialized:
                # Initialize satellite position offsets in RIC frame:
                sat_offsets_ric = np.zeros((sat_r.shape[0], 3))
                sat_offsets_ric[:, 0] = np.random.randn(sat_r.shape[0]) * sat_intrack_std
                sat_offsets_ric[:, 1] = np.random.randn(sat_r.shape[0]) * sat_crosstrack_std
                sat_offsets_ric[:, 2] = np.random.randn(sat_r.shape[0]) * sat_radial_std
                sat_offsets_initialized = True

            # Reject any satellites that are NaN:
            valid_sat = ~np.isnan(sat_r).any(axis=1)
            sat_r = sat_r[valid_sat]
            sat_v = sat_v[valid_sat]
            sat_offsets_ric_use = sat_offsets_ric[valid_sat]

            # Apply RIC offsets to satellite positions:
            sat_r_true = apply_ric_offsets(sat_r, sat_v, sat_offsets_ric_use)

            # This assumes the true position of the satellite deviates from the catalog position by a constant RIC offset
            sat_pix, valid = camera.project_points(sat_r_true)
            sat_meas = sat_pix + np.random.normal(0, pixel_noise_std, size=sat_pix.shape)
            sat_r_catalog = sat_r[valid]
            sat_predict, _ = camera.project_points(sat_r_catalog)

        else:
            curr_star_meas_pix = np.empty((0, 2))
            curr_star_true = np.empty((0, 3))

            sat_meas = np.empty((0, 2))
            sat_predict = np.empty((0, 2))
            sat_r_catalog = np.empty((0, 3))
            sat_r_true = np.empty((0, 3))


        time = time + Time(dt, "second")

        # Simulate Gyro measurements (every step):
        gyro_bias[i+1] = gyro_bias[i] + sig_rrw * np.sqrt(dt) * np.random.randn()
        measured_rate[i+1] = w + gyro_bias[i+1] + sig_arw / np.sqrt(dt) * np.random.randn()

        # Run the MEKF:
        X_hat[i+1], P, q_hat[i+1] = MEKF_position(
            X_hat[i], P, q_hat[i], dt, meas_std, Q, 
            curr_star_meas_pix,
            curr_star_true,
            sat_meas,
            sat_r_catalog,
            measured_rate[i+1], 
            ax, ay, u0, v0
        )
        sig3[i+1] = 3 * np.sqrt(np.diag(P))

        # Propagate the truth:
        X[i+1] = propagate(X[i], dt)

        if camera_available and animate:
            star_plt.set_offsets(curr_star_meas_pix)
            sat_plt.set_offsets(sat_meas)
            sat_predict_plt.set_offsets(sat_predict)
            fig.canvas.draw_idle()
            plt.pause(0.1)

    if animate:
        plt.ioff()
        plt.close(fig)


    # Plot the results:
    t_array = np.arange(0, duration + dt, dt).flatten()

    plt.figure()
    plt.plot(
        t_array,
        camera_available_history.astype(int),
        drawstyle="steps-post",
        linewidth=0.6,
    )
    plt.title("Camera Availability Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Camera Available (True/False)")

    # Plot Position Residuals:
    position_error = X_hat[:, 0:3] - cam_pos0
    position_sig3 = sig3[:, 0:3]
    plt.figure()
    plt.subplot(3, 1, 1)
    filter_plot(t_array, position_error[:, 0], position_sig3[:, 0], "Position X Error (m)", scale = 5)
    plt.title("Position Estimation Errors")

    plt.subplot(3, 1, 2)
    filter_plot(t_array, position_error[:, 1], position_sig3[:, 1], "Position Y Error (m)", scale = 5)

    plt.subplot(3, 1, 3)
    filter_plot(t_array, position_error[:, 2], position_sig3[:, 2], "Position Z Error (m)", scale = 5)

    
    # Plot Attitude Residuals:
    q_true = X[:, 0:4]
    q_hat_sig3 = sig3[:, 3:6]
    plot_error_angles(t_array, q_hat, q_true, q_hat_sig3, scale=1.5)


    # Plot Bias Residuals:
    bias_error = np.rad2deg(X_hat[:, 6:9] - gyro_bias)
    bias_sig3 = np.rad2deg(sig3[:, 6:9])

    plt.figure()
    plt.subplot(3, 1, 1)
    filter_plot(t_array, bias_error[:, 0], bias_sig3[:, 0], "Bias X Error (deg/s)", scale=1.5)
    plt.title("Gyro Bias Estimation Errors")

    plt.subplot(3, 1, 2)
    filter_plot(t_array, bias_error[:, 1], bias_sig3[:, 1], "Bias Y Error (deg/s)", scale=1.5)

    plt.subplot(3, 1, 3)
    filter_plot(t_array, bias_error[:, 2], bias_sig3[:, 2], "Bias Z Error (deg/s)", scale=1.5)
    plt.show()


if __name__ == "__main__":
    main()