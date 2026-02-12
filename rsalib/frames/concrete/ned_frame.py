from typing import Tuple

import numpy as np

from rsalib import Epoch, Rotation
from rsalib.frames import CustomFrame, FrameTransform, SpiceFrame, ecef_to_geodetic
from rsalib.state_vector import Position


class EarthNED(CustomFrame):
    def __repr__(self) -> str:
        return f"EarthNED(DATUM={self.get_datum()})"

    def __init__(self, origin, units=None):
        if isinstance(origin, Position):
            self.relative_origin = origin
        elif isinstance(origin, np.ndarray):
            if origin.shape != (3,):
                raise ValueError("If origin is a numpy array, it must be of shape (3,)")
            if units is None:
                raise ValueError("If origin is a numpy array, units must be provided.")
            self.relative_origin = Position(
                origin[0],
                origin[1],
                origin[2],
                units=units,
                frame=SpiceFrame("EARTH", "ITRF93"),
            )
        else:
            raise TypeError(
                "origin must be a Position object "
                "or a numpy array of shape (3,) representing ECEF coordinates."
            )
        self._datum = SpiceFrame("EARTH", "ITRF93")

    def get_datum(self) -> SpiceFrame:
        return self._datum

    def get_transform_to_datum(self, time: Epoch) -> Tuple[FrameTransform, SpiceFrame]:
        ecef_position = self.relative_origin.transform_to(self._datum, time)
        lat, lon, _ = ecef_to_geodetic(ecef_position.values)

        # Construct the ECEF->NED rotation matrix
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        matrix = np.array(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )

        transform = FrameTransform(
            from_frame=self,
            to_frame=self._datum,
            translation=self.relative_origin,
            rotation=Rotation(matrix),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        return transform, self._datum
