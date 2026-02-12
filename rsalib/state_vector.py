import numpy as np

from rsalib import Epoch, Rotation, units
from rsalib.frames.reference_frame import ReferenceFrame
from rsalib.vector3 import Vector3, Vector3Array


class StateVector:
    def __init__(
        self,
        reference_frame: ReferenceFrame,
        position: Vector3 = None,
        rotation: Rotation = None,
        velocity: Vector3 = None,
        angular_velocity: Vector3 = None,
    ):
        self._reference_frame = reference_frame

        self._position = (
            position
            if position is not None
            else Vector3(0, 0, 0, unit=units.Distance.base_unit())
        )
        self._rotation = rotation if rotation is not None else Rotation.identity()
        self._velocity = (
            velocity
            if velocity is not None
            else Vector3(0, 0, 0, unit=units.Velocity.base_unit())
        )
        self._angular_velocity = (
            angular_velocity
            if angular_velocity is not None
            else Vector3(0, 0, 0, unit=units.AngularVelocity.base_unit())
        )

    def transform_to(self, new_frame: ReferenceFrame, epoch: Epoch) -> "StateVector":
        from rsalib.frames.frame_transform import FrameTransform

        if new_frame == self.reference_frame:
            return self

        transform: FrameTransform = self.reference_frame.get_transform_to(
            new_frame, epoch
        )

        new_position = transform.apply_to_position(self.position)
        new_rotation = transform.apply_to_rotation(self.rotation)
        new_velocity = transform.apply_to_velocity(self.velocity)
        new_angular_velocity = transform.apply_to_angular_velocity(
            self.angular_velocity
        )

        return StateVector(
            reference_frame=new_frame,
            position=new_position,
            rotation=new_rotation,
            velocity=new_velocity,
            angular_velocity=new_angular_velocity,
        )

    @property
    def reference_frame(self) -> ReferenceFrame:
        return self._reference_frame

    @property
    def position(self) -> Vector3:
        return self._position

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @property
    def velocity(self) -> Vector3:
        return self._velocity

    @property
    def angular_velocity(self) -> Vector3:
        return self._angular_velocity

    def __repr__(self) -> str:
        return (
            f"StateVector(\n"
            f"  reference_frame={self.reference_frame},\n"
            f"  position={self.position},\n"
            f"  rotation={self.rotation.quaternion},\n"
            f"  velocity={self.velocity},\n"
            f"  angular_velocity={self.angular_velocity}\n"
            f")"
        )


class Position(Vector3):
    _unit_class = units.Distance

    def __init__(self, *args, units=None, frame: ReferenceFrame = None):
        if frame is None:
            raise ValueError("frame parameter must be provided for Position")
        super().__init__(*args, units=units)
        self._frame = frame

    def transform_to(self, new_frame: ReferenceFrame, epoch: Epoch) -> "Position":
        if new_frame == self._frame:
            return self
        transform = self._frame.get_transform_to(new_frame, epoch)
        v_transform = transform.apply_to_position(self.v)
        return Position(v_transform, units=self._unit, frame=new_frame)

    def __str__(self) -> str:
        return f"Position(x={self.x}, y={self.y}, z={self.z}, units={self._unit}, frame={self._frame})"

    def __repr__(self) -> str:
        return self.__str__()


class Velocity(Vector3):
    _unit_class = units.Velocity

    def __init__(self, *args, units=None, frame: ReferenceFrame = None):
        if frame is None:
            raise ValueError("frame parameter must be provided for Velocity")
        super().__init__(*args, units=units)
        self._frame = frame

    def transform_to(self, new_frame: ReferenceFrame, epoch: Epoch) -> "Velocity":
        if new_frame == self._frame:
            return self
        transform = self._frame.get_transform_to(new_frame, epoch)
        v_transform = transform.apply_to_velocity(self.v)
        return Velocity(v_transform, units=self._unit, frame=new_frame)

    def __str__(self) -> str:
        return f"Velocity(x={self.x}, y={self.y}, z={self.z}, unit={self._unit}, frame={self._frame})"

    def __repr__(self) -> str:
        return self.__str__()


class Attitude(Rotation):
    def __init__(self, *args, frame: ReferenceFrame):
        super().__init__(*args)
        self.frame = frame

    def transform_to(self, new_frame: ReferenceFrame, epoch: Epoch) -> "Attitude":
        if new_frame == self.frame:
            return self
        transform = self.frame.get_transform_to(new_frame, epoch)
        r_transform = transform.apply_to_rotation(self)
        return Attitude(r_transform, frame=new_frame)

    def __str__(self) -> str:
        return f"Attitude(quaternion={self.quaternion}, frame={self.frame})"

    def __repr__(self) -> str:
        return self.__str__()


class AngularVelocity(Vector3):
    _unit_class = units.AngularVelocity

    def __init__(self, *args, units=None, frame: ReferenceFrame = None):
        if frame is None:
            raise ValueError("frame parameter must be provided for AngularVelocity")
        super().__init__(*args, units=units)
        self.frame = frame

    def transform_to(
        self, new_frame: ReferenceFrame, epoch: Epoch
    ) -> "AngularVelocity":
        if new_frame == self.frame:
            return self
        transform = self.frame.get_transform_to(new_frame, epoch)
        v_transform = transform.apply_to_angular_velocity(self.v)
        return AngularVelocity(v_transform, frame=new_frame)

    def __str__(self) -> str:
        return f"AngularVelocity(x={self.x}, y={self.y}, z={self.z}, units={self._unit}, frame={self.frame})"

    def __repr__(self) -> str:
        return self.__str__()


#############################################################
### Array versions of Position, Velocity, AngularVelocity ###
#############################################################
class PositionArray(Vector3Array):
    _unit_class = units.Distance

    def __init__(self, data, frame: ReferenceFrame = None, units=None):
        if frame is None:
            raise ValueError("frame parameter must be provided for PositionArray")
        super().__init__(data, units=units)
        self._frame = frame

    def transform_to(self, new_frame, epoch):
        if new_frame == self._frame:
            return self

        # Vectorized transform if possible, else fallback to list comprehension
        transform = self.frame.get_transform_to(new_frame, epoch)
        new_data = np.stack(
            [
                transform.apply_to_position(
                    self._unit_class(v[0], v[1], v[2], unit=self._unit).v
                )
                for v in self._data
            ]
        )
        return PositionArray(new_data, unit=self._unit, frame=new_frame)

    def to_positions(self):
        # Convert to list of Position objects
        from rsalib.state_vector import Position

        return [Position(v, frame=self._frame) for v in self._data]


def VelocityArray(Vector3Array):
    _unit_class = units.Velocity

    def __init__(self, data, frame=None, unit=None):
        super().__init__(data, unit=unit, frame=frame)

    def transform_to(self, new_frame, epoch):
        if new_frame == self.frame:
            return self
        # Vectorized transform if possible, else fallback to list comprehension
        transform = self.frame.get_transform_to(new_frame, epoch)
        new_data = np.stack(
            [
                transform.apply_to_velocity(
                    self._unit_class(v[0], v[1], v[2], unit=self._unit).v
                )
                for v in self._data
            ]
        )
        return VelocityArray(new_data, frame=new_frame, unit=self._unit)

    def to_velocities(self):
        # Convert to list of Velocity objects
        from rsalib.state_vector import Velocity

        return [Velocity(v, frame=self._frame) for v in self._data]


def AngularVelocityArray(Vector3Array):
    _unit_class = units.AngularVelocity

    def __init__(self, data, frame=None, unit=None):
        super().__init__(data, unit=unit, frame=frame)

    def transform_to(self, new_frame, epoch):
        # Vectorized transform if possible, else fallback to list comprehension
        if new_frame == self.frame:
            return self

        transform = self.frame.get_transform_to(new_frame, epoch)
        new_data = np.stack(
            [
                transform.apply_to_angular_velocity(
                    self._unit_class(v[0], v[1], v[2], unit=self._unit).v
                )
                for v in self._data
            ]
        )
        return AngularVelocityArray(new_data, frame=new_frame, unit=self._unit)

    def to_angular_velocities(self):
        # Convert to list of AngularVelocity objects
        from rsalib.state_vector import AngularVelocity

        return [AngularVelocity(v, frame=self._frame) for v in self._data]
