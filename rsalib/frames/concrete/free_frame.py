from typing import Tuple

import numpy as np

from rsalib import Epoch, Rotation
from rsalib.frames import CustomFrame, FrameTransform, SpiceFrame


class FreeFrame(CustomFrame):
    def __repr__(self) -> str:
        return f"FreeFrame(DATUM={self.get_datum()}, NAME={self._name})"

    def __init__(self, datum: SpiceFrame, name: str = "FreeFrame"):
        self._datum = datum
        self._name = name

        # These are all relative to the datum frame
        self.relative_origin = np.zeros(3)
        self.relative_rotation = Rotation.identity()
        self.relative_velocity = np.zeros(3)
        self.relative_angular_velocity = np.zeros(3)

    def get_datum(self) -> SpiceFrame:
        return self._datum

    def get_transform_to_datum(self, time: Epoch) -> Tuple[FrameTransform, SpiceFrame]:
        transform = FrameTransform(
            self,
            self._datum,
            translation=self.relative_origin,
            rotation=self.relative_rotation,
            velocity=self.relative_velocity,
            angular_velocity=self.relative_angular_velocity,
        )
        return transform, self._datum
