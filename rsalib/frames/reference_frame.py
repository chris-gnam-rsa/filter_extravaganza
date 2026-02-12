from abc import ABC, abstractmethod

from rsalib import Epoch


# Refnerence frame base class
class ReferenceFrame(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    def get_transform_to(self, other: "ReferenceFrame", epoch: Epoch):
        return get_transform_between_frames(self, other, epoch)

    def get_position_relative_to(self, other: "ReferenceFrame", epoch: Epoch):
        return self.get_transform_to(other, epoch).translation

    def get_rotation_relative_to(self, other: "ReferenceFrame", epoch: Epoch):
        return self.get_transform_to(other, epoch).rotation

    def get_velocity_relative_to(self, other: "ReferenceFrame", epoch: Epoch):
        pass

    def get_angular_velocity_relative_to(self, other: "ReferenceFrame", epoch: Epoch):
        pass


def get_transform_between_frames(
    from_frame: "ReferenceFrame", to_frame: "ReferenceFrame", epoch: Epoch
):
    from rsalib.frames.custom_frame import CustomFrame
    from rsalib.frames.frame_transform import FrameTransform
    from rsalib.frames.spice_frame import (
        SpiceFrame,
        get_transform_between_spice_frames,
    )

    if from_frame == to_frame:
        return FrameTransform(from_frame, to_frame)  # Identity transform

    # Collect all possible transforms:
    T_c1_to_s1 = None
    T_s1_to_s2 = None  # This transform is gauranteed to exist
    T_s2_to_c2 = None

    if isinstance(from_frame, SpiceFrame):
        spice1 = from_frame
    elif isinstance(from_frame, CustomFrame):
        T_c1_to_s1, spice1 = from_frame.get_transform_to_datum(
            epoch
        )  # Custom -> SPICE 1
    else:
        raise TypeError("from_frame must be a SpiceFrame or CustomFrame")

    if isinstance(to_frame, SpiceFrame):
        spice2 = to_frame
    elif isinstance(to_frame, CustomFrame):
        T_s2_to_c2, spice2 = to_frame.get_transform_from_datum(
            epoch
        )  # SPICE 2 -> Custom
    else:
        raise TypeError("to_frame must be a SpiceFrame or CustomFrame")

    T_s1_to_s2 = get_transform_between_spice_frames(
        spice1, spice2, epoch
    )  # SPICE 1 -> SPICE 2

    # Combine transforms:
    transform = FrameTransform(from_frame, to_frame)  # Identity
    if T_c1_to_s1 is not None:
        T_c1_to_s2 = T_c1_to_s1.apply(T_s1_to_s2)
    else:
        T_c1_to_s2 = T_s1_to_s2

    if T_s2_to_c2 is not None:
        transform = T_c1_to_s2.apply(T_s2_to_c2)
    else:
        transform = T_c1_to_s2

    return transform
