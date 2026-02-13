from src import Camera, Rotation

import numpy as np
import matplotlib.pyplot as plt

class Tracklet:
    def __init__(self, position_uncertainty=0.5, velocity_uncertainty=50):
        self.observations = []

        # Current pixel position and velocity estimates [x, y, vx, vy]
        self.x_hat = np.array([0., 0., 0., 0.])
        
        self.P = np.diag([
            position_uncertainty**2, position_uncertainty**2, 
            velocity_uncertainty**2, velocity_uncertainty**2
            ])
        
        # H extracts [x, y] from [x, y, vx, vy]
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        
        self.R = np.diag([position_uncertainty**2, position_uncertainty**2])
        
        # Tuned process noise for constant velocity model:
        self.Q = np.diag([0, 0, 1, 1])

    def update(self, obs, dt):
        if obs.size == 0:
            return

        obs = obs.flatten()

        if len(self.observations) == 0:
            self.observations.append(obs)
            self.x_hat[:2] = obs
            self.x_hat[2:] = 0
        else:
            # Propagate FIRST
            self.Phi = np.eye(4)
            self.Phi[0, 2] = dt  
            self.Phi[1, 3] = dt        

            self.x_hat = self.Phi @ self.x_hat
            self.P = self.Phi @ self.P @ self.Phi.T + self.Q * dt**2

            # Then check gate and update
            gate = 3 * np.sqrt(np.diag(self.P)[:2])
            
            in_x = abs(obs[0] - self.x_hat[0]) <= gate[0]
            in_y = abs(obs[1] - self.x_hat[1]) <= gate[1]
            
            if in_x and in_y:
                dx = obs - self.x_hat[:2]
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x_hat = self.x_hat + K @ dx
                self.P = (np.eye(4) - K @ self.H) @ self.P

    def prediction_gate(self, dt):
        # Simple constant velocity model
        x_pred = self.x_hat[:2] + self.x_hat[2:] * dt
        
        # We need a temporary Phi here to project P correctly for the gate
        Phi_temp = np.eye(4)
        Phi_temp[0, 2] = dt
        Phi_temp[1, 3] = dt
        
        P_pred = Phi_temp @ self.P @ Phi_temp.T + self.Q * dt**2

        # Compute gate (e.g., 3-sigma ellipse)
        gate_size = 3 * np.sqrt(np.diag(P_pred)[:2])
        return x_pred, gate_size


class TrackletManager:
    def __init__(self):
        self.tracklets = []
    
    def add_observations(self, obs, dt):
        # For simplicity, we assume all observations belong to the same tracklet
        if not self.tracklets:
            self.tracklets.append(Tracklet())
        self.tracklets[0].update(obs, dt)

def main():
    ###########################
    ### Simulation Settings ###
    ###########################
    dt = 0.01                # simulation step
    camera_dt = 0.01         # camera period in seconds
    duration = 30        # total simulation duration in seconds

    animate = False

    fraction_images_unavailable = 0  # Fraction of images where camera is unavailable

    tracklet_manager = TrackletManager()

    ############################
    ### Camera Configuration ###
    ############################   
    focal_length = 50
    sensor_size = [36, 24]
    resolution = [1920, 1080]



    ####################
    ### Sensor Noise ###
    ####################
    pixel_noise_std = 0.1  # Camera centroiding noise in pixels



    ########################
    ### Pseudo Satellite ###
    ########################
    sat_r = np.array([[-70e3,-90e3,400e3]])
    sat_v = np.array([[3e3,7e3,0]])



    ##################
    ### Initialize ###
    ##################
    camera_step = int(round(camera_dt / dt))
    if camera_step < 1:
        raise ValueError("camera_dt must be >= dt")
    
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


    # Initialize:
    tsteps = int(duration / dt) + 1


    if animate:
        plt.ion()
        fig, ax_anim = plt.subplots()
        sat_plt = ax_anim.scatter([], [], s=5, c="black")
        ax_anim.set_xlim(0, camera.resolution[0])
        ax_anim.set_ylim(0, camera.resolution[1])
        ax_anim.set_aspect("equal", adjustable="box")
        ax_anim.invert_yaxis()
        ax_anim.set_frame_on(False)
        ax_anim.grid(True)
        sat_meas_history = []    
        
        predict_plt = ax_anim.scatter([], [], s=5, c="red", marker="x")

    #######################
    ### Simulation Loop ###
    #######################
    x_hat = np.zeros((tsteps, 4))
    pixel = np.zeros((tsteps, 2))
    for i in range(tsteps-1):
        camera_available = (i % camera_step) == 0 and (np.random.rand() > fraction_images_unavailable)

        if camera_available:
            sat_pix, valid = camera.project_points(sat_r)    # Get current position
            sat_meas = sat_pix + np.random.normal(0, pixel_noise_std, size=sat_pix.shape)
            
            tracklet_manager.add_observations(sat_meas, dt)  # Update with current measurement
            
            if sat_meas.size > 0:
                pixel[i] = sat_pix
                x_hat[i] = tracklet_manager.tracklets[0].x_hat

        if camera_available and animate and sat_meas.size > 0:
            if sat_meas.size > 0:
                sat_meas_history.append(sat_meas)
            if sat_meas_history:
                sat_plt.set_offsets(np.vstack(sat_meas_history))
            predict_plt.set_offsets(tracklet_manager.tracklets[0].x_hat[:2].reshape(1, -1))
            fig.canvas.draw_idle()
            plt.pause(0.1)


        sat_r = sat_r + sat_v * dt
        
    if animate:
        plt.ioff()
        plt.close(fig)

    error = x_hat[:, :2] - pixel

    print(np.mean(error[:,0]))
    print(np.mean(error[:,1]))
    plt.figure()
    plt.plot(error[:, 0], label="x error")
    plt.plot(error[:, 1], label="y error")
    plt.legend()
    plt.title("Tracklet Position Estimation Error")
    plt.xlabel("Time Step")
    plt.ylabel("Error (pixels)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()