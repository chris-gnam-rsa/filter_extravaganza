import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def xi(q):
    q1, q2, q3, q4 = q
    return np.array([
        [ q4, -q3,  q2],
        [ q3,  q4, -q1],
        [-q2,  q1,  q4],
        [-q1, -q2, -q3]
    ])