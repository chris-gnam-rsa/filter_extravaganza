import numpy as np

from rsalib import Rotation, units
from rsalib.frames import ReferenceFrame
from rsalib.utils.utils import validate_array


class FrameTransform:
    def __init__(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        translation: np.ndarray = None,
        rotation: Rotation = None,
        velocity: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ):
        self._from_frame = from_frame
        self._to_frame = to_frame

        self._translation = translation if translation is not None else np.zeros(3)
        if rotation is None:
            self._rotation = Rotation.identity()
        elif isinstance(rotation, Rotation):
            self._rotation = rotation
        else:
            self._rotation = Rotation(rotation)
        self._velocity = velocity if velocity is not None else np.zeros(3)
        self._angular_velocity = (
            angular_velocity if angular_velocity is not None else np.zeros(3)
        )

        validate_array(self._translation, 3)
        validate_array(self._velocity, 3)
        validate_array(self._angular_velocity, 3)

    @property
    def from_frame(self) -> ReferenceFrame:
        return self._from_frame

    @property
    def to_frame(self) -> ReferenceFrame:
        return self._to_frame

    def apply(self, other):
        # avoid circular imports
        from rsalib.state_vector import AngularVelocity, Attitude, Position, Velocity

        from .frame_transform import FrameTransform

        if isinstance(other, FrameTransform):
            new_translation = self.apply_to_position(other.translation)
            new_rotation = self.apply_to_rotation(other.rotation)
            new_velocity = self.apply_to_velocity(other.velocity)
            new_angular_velocity = self.apply_to_angular_velocity(
                other.angular_velocity
            )

            return FrameTransform(
                from_frame=self._from_frame,
                to_frame=other._to_frame,
                translation=new_translation,
                rotation=new_rotation,
                velocity=new_velocity,
                angular_velocity=new_angular_velocity,
            )

        elif isinstance(other, Position):
            new_v = self.apply_to_position(other.v)
            return Position(
                new_v, units=units.Distance.base_unit(), frame=self._to_frame
            )

        elif isinstance(other, Velocity):
            new_v = self.apply_to_velocity(other.v)
            return Velocity(
                new_v, units=units.Velocity.base_unit(), frame=self._to_frame
            )
        elif isinstance(other, AngularVelocity):
            new_v = self.apply_to_angular_velocity(other.v)
            return AngularVelocity(
                new_v, units=units.AngularVelocity.base_unit(), frame=self._to_frame
            )

        elif isinstance(other, Attitude):
            return Attitude(self.apply_to_rotation(other), frame=self._to_frame)

        else:
            raise TypeError(
                "FrameTransform can only apply to FrameTransform, Position, Velocity, or AngularVelocity objects"
            )

    def __mul__(self, other):
        return self.apply(other)

    def apply_to_position(self, position: np.ndarray) -> np.ndarray:
        # Accepts (3,), (3, N), or (N, 3)
        position = np.asarray(position)
        transposed = False
        if position.ndim == 1 and position.shape[0] == 3:
            # (3,) shape
            result = self._rotation @ position + self._translation
            return result
        elif position.ndim == 2 and position.shape[0] == 3:
            # (3, N) shape
            transposed = False
        elif position.ndim == 2 and position.shape[-1] == 3:
            # (N, 3) shape
            position = position.T
            transposed = True
        else:
            raise ValueError("Position must be shape (3,), (3, N), or (N, 3)")
        result = self._rotation @ position + self._translation[:, None]
        if transposed:
            result = result.T
        return result

    def apply_to_velocity(self, velocity: np.ndarray) -> np.ndarray:
        # Accepts (3,), (3, N), or (N, 3)
        velocity = np.asarray(velocity)
        transposed = False
        if velocity.ndim == 1 and velocity.shape[0] == 3:
            # (3,) shape
            result = self._rotation @ velocity + self._velocity
            return result
        elif velocity.ndim == 2 and velocity.shape[0] == 3:
            transposed = False
        elif velocity.ndim == 2 and velocity.shape[-1] == 3:
            velocity = velocity.T
            transposed = True
        else:
            raise ValueError("Velocity must be shape (3,), (3, N), or (N, 3)")
        result = self._rotation @ velocity + self._velocity[:, None]
        if transposed:
            result = result.T
        return result

    def apply_to_direction(self, direction: np.ndarray) -> np.ndarray:
        # Accepts (3,), (3, N), or (N, 3)
        direction = np.asarray(direction)
        transposed = False
        if direction.ndim == 1 and direction.shape[0] == 3:
            # (3,) shape
            result = self._rotation @ direction
            return result
        elif direction.ndim == 2 and direction.shape[0] == 3:
            transposed = False
        elif direction.ndim == 2 and direction.shape[-1] == 3:
            direction = direction.T
            transposed = True
        else:
            raise ValueError("Direction must be shape (3,), (3, N), or (N, 3)")
        result = self._rotation @ direction
        if transposed:
            result = result.T
        return result

    def apply_to_angular_velocity(self, angular_velocity: np.ndarray) -> np.ndarray:
        # Accepts (3,), (3, N), or (N, 3)
        angular_velocity = np.asarray(angular_velocity)
        transposed = False
        if angular_velocity.ndim == 1 and angular_velocity.shape[0] == 3:
            # (3,) shape
            result = self._rotation @ angular_velocity + self._angular_velocity
            return result
        elif angular_velocity.ndim == 2 and angular_velocity.shape[0] == 3:
            transposed = False
        elif angular_velocity.ndim == 2 and angular_velocity.shape[-1] == 3:
            angular_velocity = angular_velocity.T
            transposed = True
        else:
            raise ValueError("Angular velocity must be shape (3,), (3, N), or (N, 3)")
        result = self._rotation @ angular_velocity + self._angular_velocity[:, None]
        if transposed:
            result = result.T
        return result

    def apply_to_rotation(self, rotation) -> Rotation:
        if not isinstance(rotation, Rotation):
            rotation = Rotation(rotation)
        return self._rotation @ rotation

    def apply_to_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """
        Applies the frame transform to a covariance matrix.

        Args:
            covariance (np.ndarray): The covariance matrix to transform.

        Returns:
            np.ndarray: The transformed covariance matrix.
        """
        # Assumes covariance is a 3x3 matrix
        validate_array(covariance, (3, 3))

        R = self._rotation.matrix
        return R @ covariance @ R.T

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @property
    def t(self) -> np.ndarray:
        return self._translation

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @property
    def R(self) -> Rotation:
        return self._rotation

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def v(self) -> np.ndarray:
        return self._velocity

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._angular_velocity

    @property
    def w(self) -> np.ndarray:
        return self._angular_velocity

    def inverse(self) -> "FrameTransform":
        inv_rotation = self._rotation.inverse()
        inv_translation = inv_rotation @ -self._translation
        inv_velocity = inv_rotation @ -self._velocity
        inv_angular_velocity = inv_rotation @ -self._angular_velocity
        return FrameTransform(
            from_frame=self._to_frame,
            to_frame=self._from_frame,
            translation=inv_translation,
            rotation=inv_rotation,
            velocity=inv_velocity,
            angular_velocity=inv_angular_velocity,
        )

    def __repr__(self):
        return (
            f"FrameTransform(\n"
            f"  from_frame={self._from_frame},\n"
            f"  to_frame={self._to_frame},\n"
            f"  translation={self._translation},\n"
            f"  rotation={self._rotation.quaternion},\n"
            f"  velocity={self._velocity},\n"
            f"  angular_velocity={self._angular_velocity},\n"
            f")"
        )
