from abc import ABC, abstractmethod
from typing import Tuple

from rsalib import Epoch
from rsalib.frames.frame_transform import FrameTransform
from rsalib.frames.reference_frame import ReferenceFrame
from rsalib.frames.spice_frame import SpiceFrame


class CustomFrame(ReferenceFrame, ABC):
    @abstractmethod
    def get_datum(self) -> SpiceFrame:
        pass

    @abstractmethod
    def get_transform_to_datum(self, time: Epoch) -> Tuple[FrameTransform, SpiceFrame]:
        pass

    def get_transform_from_datum(
        self, time: Epoch
    ) -> Tuple[FrameTransform, SpiceFrame]:
        transform, datum = self.get_transform_to_datum(time)
        return transform.inverse(), datum
