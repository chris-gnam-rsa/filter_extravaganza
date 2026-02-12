# import numpy for array handling

from typing import Tuple

import numpy as np

from rsalib import units
from rsalib.utils.rotation_utils import (
    _axis_angle_to_quat,
    _euler_angles_to_quat,
    _matrix_to_quat,
    _mrp_to_quat,
    _quat_to_axis_angle,
    _quat_to_euler_angles,
    _quat_to_matrix,
    _quat_to_mrp,
    _quat_to_rdt,
    _rdt_to_quat,
    _vectors_to_quat,
)
from rsalib.vector3 import Vector3


class Rotation:
    """
    Rotation class using quaternion as core representation.

    This class provides a unified interface for representing and manipulating 3D rotations.
    It supports initialization from quaternions, rotation matrices, axis-angle pairs, and Euler angles.

    Supported input formats:
        - Quaternion: 4-element array/list/tuple (x, y, z, w)
        - Rotation matrix: 3x3 array/list/np.ndarray
        - Axis-angle: (axis, angle) tuple/list
        - Euler angles: (angles, sequence) tuple/list
    """

    def __init__(self, data=None, sequence=None):
        """
        Initialize a Rotation object from various input formats.

        Args:
            data: Input data representing the rotation. Can be a quaternion (4,),
                  rotation matrix (3,3), axis-angle ((3,), float), or Euler angles (3,).
            sequence: (Optional) String specifying the Euler angle sequence (e.g., 'xyz').
        Raises:
            TypeError: If the input format is not recognized.
            ValueError: If Euler angles are provided without a sequence.
        """
        if data is None:
            self.make_identity()
            return

        # Detect input type
        arr = np.asarray(data)
        if arr.shape == (4,):
            # Assume scalar-last input
            self._quat = arr.astype(float)

        elif arr.shape == (3, 3):
            self.set_from_matrix(arr)

        elif arr.shape == (2,) and isinstance(data[0], (list, tuple, np.ndarray)):
            # Axis-angle
            axis, angle = data
            self.set_from_axis_angle(axis, angle)

        elif arr.shape == (3,):
            if sequence is None:
                raise ValueError("Euler angles require a sequence string, e.g. 'xyz'")
            self.set_from_euler_angles(arr, sequence)

        else:
            raise TypeError("Unrecognized rotation input format.")

    @classmethod
    def identity(cls):
        """
        Create an identity rotation (no rotation).
        Returns:
            Rotation: Identity rotation object.
        """
        return cls([0.0, 0.0, 0.0, 1.0])

    def set_identity(self):
        """
        Set this rotation to the identity (no rotation).
        """
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])

    @classmethod
    def from_quat(cls, quat):
        """
        Create a rotation from a quaternion (scalar-last convention).
        Args:
            quat: Array-like of shape (4,) [x, y, z, w].
        Returns:
            Rotation: Rotation object.
        """
        return cls(np.asarray(quat).astype(float))

    def set_from_quat(self, quat):
        """
        Set this rotation from a quaternion (scalar-last convention).
        Args:
            quat: Array-like of shape (4,) [x, y, z, w].
        """
        self._quat = np.asarray(quat).astype(float)

    @classmethod
    def from_matrix(cls, matrix):
        """
        Create a rotation from a 3x3 rotation matrix.
        Args:
            matrix: 3x3 array-like.
        Returns:
            Rotation: Rotation object.
        """
        quat = _matrix_to_quat(matrix)
        return cls(quat)

    def set_from_matrix(self, matrix):
        """
        Set this rotation from a 3x3 rotation matrix.
        Args:
            matrix: 3x3 array-like.
        """
        quat = _matrix_to_quat(matrix)
        self._quat = quat

    @classmethod
    def from_vectors(cls, v1: np.ndarray, v2: np.ndarray):
        """
        Create a rotation that rotates vector v1 to align with vector v2 using the Rodrigues rotation formula.
        Args:
            v1 (np.ndarray): Initial vector.
            v2 (np.ndarray): Target vector.
        Returns:
            Rotation: Rotation object that rotates v1 to align with v2.
        """
        rotation_quat = _vectors_to_quat(v1, v2)
        return cls(rotation_quat)

    def set_from_vectors(self, v1: np.ndarray, v2: np.ndarray):
        """
        Set this rotation to rotate v1 to align with v2 using the Rodrigues rotation formula.
        Args:
            v1 (np.ndarray): Initial vector.
            v2 (np.ndarray): Target vector.
        """
        rotation_quat = _vectors_to_quat(v1, v2)
        self._quat = rotation_quat

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: units.Angle):
        """
        Create a rotation from an axis and angle.
        Args:
            axis: Array-like, rotation axis (3,).
            angle: Rotation angle in radians.
        Returns:
            Rotation: Rotation object.
        """
        quat = _axis_angle_to_quat(axis, angle)
        return cls(quat)

    def set_from_axis_angle(self, axis: np.ndarray, angle: units.Angle):
        """
        Set this rotation from an axis and angle.
        Args:
            axis: Array-like, rotation axis (3,).
            angle: Rotation angle in radians.
        """
        quat = _axis_angle_to_quat(axis, angle)
        self._quat = quat

    @classmethod
    def from_euler_angles(
        cls,
        axis1: units.Angle,
        axis2: units.Angle,
        axis3: units.Angle,
        sequence: str = "ZYX",
    ):
        """
        Create a rotation from Euler angles.

        Args:
            axis1 (units.Angle): First Euler angle (in radians).
            axis2 (units.Angle): Second Euler angle (in radians).
            axis3 (units.Angle): Third Euler angle (in radians).
            sequence (str, optional): Rotation order, e.g. 'ZYX'. Defaults to 'ZYX'.

        Returns:
            Rotation: Rotation object representing the specified Euler angles.
        """
        quat = _euler_angles_to_quat(axis1, axis2, axis3, sequence=sequence)
        return cls(quat)

    def set_from_euler_angles(
        self,
        axis1: units.Angle,
        axis2: units.Angle,
        axis3: units.Angle,
        sequence: str = "ZYX",
    ):
        """
        Set this rotation from Euler angles.

        Args:
            axis1 (units.Angle): First Euler angle (in radians).
            axis2 (units.Angle): Second Euler angle (in radians).
            axis3 (units.Angle): Third Euler angle (in radians).
            sequence (str, optional): Rotation order, e.g. 'ZYX'. Defaults to 'ZYX'.
        """
        quat = _euler_angles_to_quat(axis1, axis2, axis3, sequence=sequence)
        self._quat = quat

    @classmethod
    def from_mrp(cls, mrp):
        """
        Create a rotation from Modified Rodrigues Parameters (not implemented).
        Args:
            mrp: Array-like, MRP parameters.
        Raises:
            NotImplementedError
        """
        quat = _mrp_to_quat(mrp)
        return cls(quat)

    def set_from_mrp(self, mrp):
        """
        Set this rotation from Modified Rodrigues Parameters (not implemented).
        Args:
            mrp: Array-like, MRP parameters.
        Raises:
            NotImplementedError
        """
        self._quat = _mrp_to_quat(mrp)

    @classmethod
    def from_rdt(cls, ra: units.Angle, dec: units.Angle, twist: units.Angle):
        """
        Create a rotation from right ascension, declination, twist (RDT) parameters.
        Args:
            ra: Right ascension angle in radians.
            dec: Declination angle in radians.
            twist: Twist angle in radians.
        Returns:
            Rotation: Rotation object.
        """
        quat = _rdt_to_quat(ra, dec, twist)
        return cls(quat)

    def set_from_rdt(self, ra: units.Angle, dec: units.Angle, twist: units.Angle):
        """
        Set this rotation from right ascension, declination, twist (RDT) parameters.
        Args:
            ra: Right ascension angle in radians.
            dec: Declination angle in radians.
            twist: Twist angle in radians.
        """
        quat = _rdt_to_quat(ra, dec, twist)
        self._quat = quat

    @property
    def quaternion(self) -> np.ndarray:
        """
        Get the quaternion representation of the rotation (scalar-last: [x, y, z, w]).
        Returns:
            np.ndarray: Quaternion (x, y, z, w).
        """
        return self._quat

    @property
    def positive_scalar_quaternion(self) -> np.ndarray:
        """
        Return the quaternion with the scalar (w) part positive (scalar-last convention).
        If w < 0, returns -q. Otherwise, returns q unchanged.
        This is useful for enforcing a unique quaternion representation
        (since q and -q represent the same rotation).
        Returns:
            np.ndarray: Quaternion (x, y, z, w) with w >= 0.
        """
        q = self._quat
        if q[3] < 0:
            return -q
        return q

    @property
    def matrix(self) -> np.ndarray:
        """
        Get the 3x3 rotation matrix representation.
        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        return _quat_to_matrix(self._quat)

    @property
    def dcm(self) -> np.ndarray:
        """
        Get the 3x3 direction cosine matrix (DCM) representation.
        Returns:
            np.ndarray: 3x3 DCM.
        """
        return self.matrix

    @property
    def axis_angle(self) -> tuple[np.ndarray, float]:
        """
        Get the axis-angle representation of the rotation (not implemented).
        Returns:
            tuple: (axis, angle)
        Raises:
            NotImplementedError
        """
        return _quat_to_axis_angle(self._quat)

    @property
    def euler_angles(self) -> Tuple[units.Angle, units.Angle, units.Angle]:
        """
        Get the Euler angles (ZYX order) from the rotation matrix.
        Returns:
            np.ndarray: Euler angles (3,).
        """
        return _quat_to_euler_angles(self._quat, sequence="ZYX")

    @property
    def mrp(self) -> np.ndarray:
        """
        Get the Modified Rodrigues Parameters (not implemented).
        Returns:
            np.ndarray: MRP parameters.
        Raises:
            NotImplementedError
        """
        return _quat_to_mrp(self._quat)

    @property
    def rdt(self) -> Tuple[units.Angle, units.Angle, units.Angle]:
        """
        Right ascension, declination, twist representation

        Returns:
            Tuple[units.Angle, units.Angle, units.Angle]: [ra, dec, twist]
        """
        return _quat_to_rdt(self._quat)

    @property
    def x_axis(self) -> np.ndarray:
        """
        Get the rotated x-axis as a Vector3.
        Returns:
            Vector3: Rotated x-axis.
        """
        return self.matrix[:, 0]

    @property
    def y_axis(self) -> np.ndarray:
        """
        Get the rotated y-axis as a Vector3.
        Returns:
            Vector3: Rotated y-axis.
        """
        return self.matrix[:, 1]

    @property
    def z_axis(self) -> np.ndarray:
        """
        Get the rotated z-axis as a Vector3.
        Returns:
            Vector3: Rotated z-axis.
        """
        return self.matrix[:, 2]

    def __repr__(self) -> str:
        """
        String representation of the Rotation object.
        Returns:
            str: String showing the quaternion.
        """
        return f"Rotation(quat=\n{self._quat})"

    def inverse(self) -> "Rotation":
        """
        Get the inverse (transpose) of this rotation.
        Returns:
            Rotation: Inverse rotation.
        """
        # TODO Implement this to use quaternion directly:
        return Rotation(self.matrix.T)

    def apply_to_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply this rotation to a 3D vector.
        Args:
            vector (np.ndarray): 3D vector to rotate.
        Returns:
            np.ndarray: Rotated 3D vector.
        """
        return self.matrix @ vector

    def apply_to_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """
        Apply this rotation to a 3x3 covariance matrix.
        Args:
            covariance (np.ndarray): 3x3 covariance matrix.
        Returns:
            np.ndarray: Rotated covariance matrix.
        """
        return self.matrix @ covariance @ self.matrix.T

    def __mul__(self, other):
        """
        Multiplication operator (*) for Rotation objects.
        Supports multiplication with another Rotation, a 3D vector, or a StateVector.
        Args:
            other: Rotation, np.ndarray (3,), or StateVector.
        Returns:
            Rotation, np.ndarray, or StateVector.
        Raises:
            ValueError: If multiplying by a vector of incorrect shape.
            TypeError: If the operand type is not supported.
        """
        return self.__matmul__(other)

    def __matmul__(self, other):
        """
        Matrix multiplication operator (@) for Rotation objects.
        Supports multiplication with another Rotation, a 3D vector, or a StateVector.
        Args:
            other: Rotation, np.ndarray (3,), or StateVector.
        Returns:
            Rotation, np.ndarray, or StateVector.
        Raises:
            ValueError: If multiplying by a vector of incorrect shape.
            TypeError: If the operand type is not supported.
        """
        # TODO Implement quaternion multiplication directly
        if isinstance(other, Rotation):
            return Rotation(self.matrix @ other.matrix)

        if isinstance(other, np.ndarray):
            if other.shape == (3,):
                return self.matrix @ other
            elif other.ndim == 2 and other.shape[0] == 3:
                # (3, N) shape: matrix @ array
                return self.matrix @ other
            elif other.ndim == 2 and other.shape[1] == 3:
                # (N, 3) shape: transpose, multiply, transpose back
                return (self.matrix @ other.T).T
            else:
                raise ValueError(
                    "Can only multiply Rotation by a 3D vector or array of shape (3,), (3,N), or (N,3)."
                )

        if isinstance(other, Vector3):
            return Vector3(self.matrix @ other.v)

        # Import StateVector only when needed to avoid circular import
        from rsalib.state_vector import (
            AngularVelocity,
            Attitude,
            Position,
            StateVector,
            Velocity,
        )

        if isinstance(other, StateVector):
            # Apply rotation to position and velocity components
            rotated_position = self.matrix @ other.position
            rotated_orientation = self @ other.rotation
            rotated_velocity = self.matrix @ other.velocity
            rotated_angular_velocity = self.matrix @ other.angular_velocity
            return StateVector(
                position=rotated_position,
                rotation=rotated_orientation,
                velocity=rotated_velocity,
                angular_velocity=rotated_angular_velocity,
            )

        if isinstance(other, Position):
            return Position(self.matrix @ other.v, frame=other.frame)

        if isinstance(other, Velocity):
            return Velocity(self.matrix @ other.v, frame=other.frame)

        if isinstance(other, Attitude):
            return Attitude(self @ other, frame=other.frame)

        if isinstance(other, AngularVelocity):
            return AngularVelocity(self.matrix @ other.v, frame=other.frame)

        # If we get to the end, its invalid:
        raise TypeError(
            "Rotation can only be multiplied by another Rotation, a 3D vector, or a StateVector."
        )
