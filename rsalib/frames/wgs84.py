from typing import Tuple

import numpy as np

from rsalib.state_vector import Position, units

# SPICE WGS84 constants
WGS84 = {
    "SemiMajorAxis": 6378.1366 * units.KM,  # SPICE WGS84 semi-major axis in meters
    "SemiMinorAxis": 6356.7519 * units.KM,  # SPICE WGS84 semi-minor axis in meters
    "Flattening": 0.0033528131084554717,  # SPICE WGS84 flattening
}


def ecef_to_geodetic(
    position: np.array,
) -> Tuple[units.Angle, units.Angle, units.Distance]:
    """
    Convert ECEF coordinates to geodetic latitude, longitude, and altitude.
    This is a simplified version and assumes a spherical Earth for demonstration purposes.

    Args:
        position (np.array): ECEF position as a numpy array in units of meters

    Returns:
        Tuple[units.Angle, units.Angle, units.Distance]: Latitude, Longitude, Altitude
    """

    x = position[0]
    y = position[1]
    z = position[2]

    a = WGS84["SemiMajorAxis"].to_float("meter")
    b = WGS84["SemiMinorAxis"].to_float("meter")
    e2 = 1 - (b**2 / a**2)

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(z + (e2 * b) * sin_theta**3, p - (e2 * a) * cos_theta**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt


def geodetic_to_ecef(
    lat: units.Angle, lon: units.Angle, alt: units.Distance
) -> Position:
    """
    Convert geodetic latitude, longitude, and altitude to ECEF coordinates.
    This is a simplified version and assumes a spherical Earth for demonstration purposes.
    """
    from rsalib.frames import SpiceFrame

    alt = alt.to_float("meter")
    a = WGS84["SemiMajorAxis"].to_float("meter")
    f = WGS84["Flattening"]
    e2 = 2 * f - f**2

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return Position(x, y, z, units=units.METER, frame=SpiceFrame("EARTH", "ITRF93"))
