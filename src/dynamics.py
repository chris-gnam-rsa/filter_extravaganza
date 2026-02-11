import numpy as np

from src.math import skew

def rk4(dynamics, dt, X, *args):
    k1 = dt * dynamics(0,       X,          *args)
    k2 = dt * dynamics(0.5*dt,  X + 0.5*k1, *args)
    k3 = dt * dynamics(0.5*dt,  X + 0.5*k2, *args)
    k4 = dt * dynamics(dt,      X + k3,     *args)
    X_next = X + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return X_next

def attdyn(_, X, J, dw_input):
    # Extract and normalize quaternion
    q = X[:4]
    q = q / np.linalg.norm(q)
    # Extract angular velocity
    w = X[4:]

    # Calculate angular momentum dynamics
    dw = np.linalg.solve(J, np.cross(-w, J @ w))
    dw = dw + dw_input # Account for input accelerations

    # Quaternion kinematics (Bq matrix)
    Bq = np.zeros((4,3))
    Bq[:3,:] = skew(q[:3]) + np.diag([q[3], q[3], q[3]])
    Bq[3,:] = -q[:3]
    dq = 0.5 * Bq @ w

    # Combine into a state derivative
    dX = np.concatenate((dq, dw))
    return dX


def propagate(X, dt, J=np.eye(3), dw_input = np.zeros(3)):
    X = rk4(attdyn, dt, X, J, dw_input)
    X[:4] = X[:4] / np.linalg.norm(X[:4])
    return X

def propagate_quaternion(q, omega, dt):
    # Quaternion kinematics (Bq matrix)
    Bq = np.zeros((4,3))
    Bq[:3,:] = skew(q[:3]) + np.diag([q[3], q[3], q[3]])
    Bq[3,:] = -q[:3]
    dq = 0.5 * Bq @ omega
    q_next = q + dq * dt
    return q_next / np.linalg.norm(q_next)

def cov_dot(P, F, G, Qc):
    # P is 6x6
    return F @ P @ F.T + G @ Qc @ G.T

def propagate_covariance(P, F, G, Qc, dt):
    # Use RK4 integration for accuracy, like your state
    def dyn(_, P_flat):
        P = P_flat.reshape(F.shape)
        dP = cov_dot(P, F, G, Qc)
        return dP.flatten()
    P_flat = P.flatten()
    P_flat_next = rk4(dyn, dt, P_flat)
    return P_flat_next.reshape(P.shape)