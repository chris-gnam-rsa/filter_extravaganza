import numpy as np

def earth_rotation_matrix(dt: float) -> np.ndarray:
    # Earth's rotation rate (rad/s)
    omega_earth = 7.2921150e-5  
    theta = omega_earth * dt
    R_earth = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,              0,             1]])
    return R_earth

def lla(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    # Convert latitude, longitude, altitude to ECEF coordinates
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    a = 6378137.0  # Semi-major axis (WGS84)
    b = 6356752.3142  # Semi-minor axis (WGS84)
    e2 = 1 - (b**2 / a**2)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)

    return np.array([x, y, z])