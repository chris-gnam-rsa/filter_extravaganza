import numpy as np
import spiceypy as spice

from rsalib import Epoch, Rotation
from rsalib.frames.frame_transform import FrameTransform
from rsalib.frames.reference_frame import ReferenceFrame
from rsalib.utils.spice_utils import (
    _load_required_kernels,
    _normalize_body,
    _normalize_frame,
    load_kernels,
)


def get_transform_between_spice_frames(
    from_frame: "SpiceFrame", to_frame: "SpiceFrame", epoch: Epoch
) -> FrameTransform:
    # Get relative position/velocity between frames:
    state, lt = spice.spkezr(
        from_frame.origin_name,
        epoch.et,
        to_frame.frame_name,
        "NONE",
        to_frame.origin_name,
    )
    position = np.array(state[:3])
    velocity = np.array(state[3:6])

    # Get rotation matrix from SPICE:
    rotmat = spice.pxform(from_frame.frame_name, to_frame.frame_name, epoch.et)
    rotation = Rotation(rotmat)

    # TODO Compute angular velocity between frames:
    angular_velocity = np.zeros(3)

    # Create Transform object:
    transform = FrameTransform(
        from_frame,
        to_frame,
        translation=position,
        rotation=rotation,
        velocity=velocity,
        angular_velocity=angular_velocity,
    )
    return transform


class SpiceFrame(ReferenceFrame):
    def __repr__(self) -> str:
        return f"SpiceFrame(ORIGIN='{self._origin_name}', ORIENTATION='{self._orientation}')"

    def __init__(self, origin, orientation, custom_kernels=None):
        self._origin_id, self._origin_name = _normalize_body(origin)
        self._orientation = _normalize_frame(orientation)

        load_kernels(custom_kernels)
        _load_required_kernels(self._origin_id, self._origin_name, self._orientation)

    @property
    def origin_name(self) -> str:
        return self._origin_name

    @property
    def frame_name(self) -> str:
        return self._orientation


def ECIJ2000(custom_kernels=None) -> SpiceFrame:
    """Create an Earth-Centered Inertial (ECI) J2000 SpiceFrame."""
    return SpiceFrame("EARTH", "J2000", custom_kernels=custom_kernels)


def ECEFITRF93(custom_kernels=None) -> SpiceFrame:
    """Create an Earth-Centered Earth-Fixed (ECEF) ITRF93 SpiceFrame."""
    return SpiceFrame("EARTH", "ITRF93", custom_kernels=custom_kernels)


def get_eci_to_ecef(epoch: Epoch, custom_kernels=None) -> FrameTransform:
    """Get the transform from ECI J2000 to ECEF ITRF93 at the given epoch."""
    eci_frame = ECIJ2000(custom_kernels=custom_kernels)
    ecef_frame = ECEFITRF93(custom_kernels=custom_kernels)
    return eci_frame.get_transform_to(ecef_frame, epoch)
