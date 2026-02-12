import numpy as np

# Unit Conversions
KM_TO_M = 1000  # m/km
M_TO_KM = KM_TO_M**-1  # km/m

DAY_TO_SEC = 86400  # s/day
SEC_TO_DAY = DAY_TO_SEC**-1  # day/s

ARCSECONDS_TO_DEGREES = 1 / 3600  # degrees per arcsecond

M_CAMERA2IMAGE = np.eye(3)  # Camera-to-Image Frame transformation matrix
M_ICRF2IMAGE_ZERO_RDT = np.array(
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
)  # ICRF-to-Image Frame at zero right ascension, declination, and twist about boresight
V_BORESIGHT_IMAGE_FRAME = np.array([0, 0, 1])  # Boresight vector in Image Frame

SPEED_OF_LIGHT_M_PER_S = 299792458  # Speed of light in m/s

# Software defined limits
CATALOG_MIN_TARGET_ALTITUDE_KM = 170.0
CATALOG_MAX_TARGET_ALTITUDE_KM = 5000.0

ALBEDO = 0.2  # Default albedo for satellites - derivded from  "Comparing Photometric Behavior of LEO Constellations to SpaceX Starlink using a space-
# based optical sensor" C. Johnson et. al, 2021
SOLAR_FLUX = 1361.0  # Solar flux at 1 AU in W/m²
V_BAND_ZERO_POINT = 3.631e-9  # Johnson V-band zero point in W/m²

# Orbital mechanics
G = 6.67430e-11  # Nm^2/kg^2
G1 = 9.80665  # m/s^2
M_EARTH = 5.9722e24  # kg
MU_EARTH = 3.986004418e14  # m^3/s^2
