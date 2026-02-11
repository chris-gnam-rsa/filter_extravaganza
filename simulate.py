from src import create_stars, Camera, Rotation
from src.dynamics import propagate
from estimators import startracker

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Simulation Settings:
    num_stars = 10000
    tsteps = 1000
    dt = 1

    # Initial states:
    w0 = np.array([0.1, 0.1, 0.1]) * np.pi / 180  # rad/s
    q0 = np.array([0, 0, 0, 1])


    # Measurement noise:
    pixel_noise_std = 1.0  # pixels


    # Create camera:
    camera = Camera(
        position=[0, 0, 0],
        orientation=Rotation.from_euler([0,0,0], order="xyz", degrees=True),
        focal_length=50,
        sensor_size=[36, 24],
        resolution=[1920, 1080],
    )

    # Create the stars:
    stars = create_stars(num_stars)

    # Simulate camera motion:
    

    # Initialize:
    X = np.zeros((tsteps, 4+3))
    X[0, 0:4] = q0
    X[0, 4:7] = w0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter([], [], s=6, color="white", marker="o")
    scatter_star_meas = ax.scatter([], [], s=10, color="red", alpha=0.5, marker="x")
    ax.set_xlim(0, camera.resolution[0])
    ax.set_ylim(0, camera.resolution[1])
    ax.set_facecolor("black")
    ax.set_title(f"Projection of {num_stars} Stars")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_aspect("equal", adjustable="box")

    for i in range(tsteps):
        # Propagate truth:
        X[i+1] = propagate(X[i], dt)
        q = X[i+1, 0:4]

        camera.orientation = Rotation.from_quaternion(q)

        star_pix, valid = camera.project_directions(stars)
        star_meas = star_pix + np.random.normal(0, pixel_noise_std, size=star_pix.shape)
        star_meas_vec = camera.pixel_to_rays_body(star_meas)

        R_opt, R_cov_opt = startracker(
            detected_stars=star_meas_vec,
            vec_meas=stars[valid],
            meas_uncertainty=np.array([pixel_noise_std, pixel_noise_std]),
            focal_length_pix=camera.focal_length,
        )

        print(q)
        print(Rotation.from_matrix(R_opt).quaternion)
        print("\n")

        scatter.set_offsets(star_pix)
        scatter_star_meas.set_offsets(star_meas)
        fig.canvas.draw_idle()
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()