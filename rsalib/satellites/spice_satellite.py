import spiceypy as spice

from rsalib import Epoch, units
from rsalib.frames import SpiceFrame
from rsalib.state_vector import AngularVelocity, Attitude, Position, Velocity


class SpiceSatellite:
    """SpiceSatellite class using SPICE kernels for satellite position and velocity."""

    def __init__(self, spice_name: str, spice_frame: str) -> None:
        """SpiceSatellite constructor.

        Args:
            spice_name (str): SPICE satellite name.
            spice_frame (str): SPICE satellite reference frame.
        """
        self._spice_name = spice_name
        self._spice_frame = spice_frame

    def get_state(
        self, epoch: Epoch, observer_frame: SpiceFrame, abcorr: str = "NONE"
    ) -> tuple[Position, Velocity]:
        """Get the satellite position and velocity at a given epoch."""
        state, lt = spice.spkezr(
            self._spice_name,
            epoch.et,
            observer_frame.frame_name,
            abcorr,
            observer_frame.origin_name,
        )
        position = Position(state[:3] * units.KM, frame=observer_frame)
        velocity = Velocity(state[3:6] * units.KPS, frame=observer_frame)
        return position, velocity

    def get_attitude(
        self, epoch: Epoch, observer_frame: SpiceFrame
    ) -> tuple[Attitude, AngularVelocity]:
        """Get the satellite attitude and angular velocity at a given epoch."""
        # Get the transformation matrix from the satellite frame to the observer frame
        xform = spice.sxform(self._spice_frame, observer_frame.frame_name, epoch.et)
        matrix, av = spice.xf2rav(xform)
        attitude = Attitude(matrix, frame=observer_frame)
        angular_velocity = AngularVelocity(av, units.RAD_PER_S, frame=observer_frame)
        return attitude, angular_velocity
