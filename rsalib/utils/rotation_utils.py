from typing import Tuple

import numpy as np

from rsalib import constants, units
from rsalib.utils.math import skew_symmetric


# Quaternion conversion utilities (scalar-last <-> scalar-first)
def _to_scalar_first(q: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternion [x, y, z, w] to scalar-first [w, x, y, z].

    Args:
        q (np.ndarray): Quaternion in scalar-last format [x, y, z, w].

    Returns:
        np.ndarray: Quaternion in scalar-first format [w, x, y, z].
    """
    q = np.asarray(q)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _from_scalar_first(q: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion [w, x, y, z] to scalar-last [x, y, z, w].

    Args:
        q (np.ndarray): Quaternion in scalar-first format [w, x, y, z].

    Returns:
        np.ndarray: Quaternion in scalar-last format [x, y, z, w].
    """
    q = np.asarray(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion (scalar-last).

    Args:
        matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """
    q = np.zeros(4)
    tr = np.trace(matrix)

    if tr > 0:
        sqtrp1 = np.sqrt(tr + 1.0)

        q[0] = sqtrp1 / 2
        q[1] = (matrix[1, 2] - matrix[2, 1]) / (2 * sqtrp1)
        q[2] = (matrix[2, 0] - matrix[0, 2]) / (2 * sqtrp1)
        q[3] = (matrix[0, 1] - matrix[1, 0]) / (2 * sqtrp1)
    else:
        d = np.diag(matrix)
        if d[1] > d[0] and d[1] > d[2]:
            # max value at self.m[1,1]
            sqdip1 = np.sqrt(d[1] - d[0] - d[2] + 1)
            q[2] = sqdip1 / 2
            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1
            q[0] = (matrix[2, 0] - matrix[0, 2]) * sqdip1
            q[1] = (matrix[0, 1] + matrix[1, 0]) * sqdip1
            q[3] = (matrix[1, 2] + matrix[2, 1]) * sqdip1
        elif d[2] > d[0]:
            # max value at self.m[2,2]
            sqdip1 = np.sqrt(d[2] - d[0] - d[1] + 1)
            q[3] = 0.5 * sqdip1
            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1
            q[0] = (matrix[0, 1] - matrix[1, 0]) * sqdip1
            q[1] = (matrix[2, 0] + matrix[0, 2]) * sqdip1
            q[2] = (matrix[1, 2] + matrix[2, 1]) * sqdip1
        else:
            # max value at self.m[0,0]
            sqdip1 = np.sqrt(d[0] - d[1] - d[2] + 1)
            q[1] = 0.5 * sqdip1
            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1
            q[0] = (matrix[1, 2] - matrix[2, 1]) * sqdip1
            q[2] = (matrix[0, 1] + matrix[1, 0]) * sqdip1
            q[3] = (matrix[2, 0] + matrix[0, 2]) * sqdip1

    return _from_scalar_first(q)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Converts the quaternion to a direction cosine matrix (DCM).

    Returns:
        R: The direction cosine matrix.

    References:
        W.G. Breckenridge IOM 343-79-1199, "Quaternions - Proposed Standard Conventions".
    """

    matrix = np.empty((3, 3))

    matrix[0, 0] = 2 * (quat[3] ** 2 + quat[0] ** 2) - 1
    matrix[0, 1] = 2 * (quat[0] * quat[1] + quat[3] * quat[2])
    matrix[0, 2] = 2 * (quat[0] * quat[2] - quat[3] * quat[1])

    matrix[1, 0] = 2 * (quat[0] * quat[1] - quat[3] * quat[2])
    matrix[1, 1] = 2 * (quat[3] ** 2 + quat[1] ** 2) - 1
    matrix[1, 2] = 2 * (quat[1] * quat[2] + quat[0] * quat[3])

    matrix[2, 0] = 2 * (quat[0] * quat[2] + quat[3] * quat[1])
    matrix[2, 1] = 2 * (quat[1] * quat[2] - quat[0] * quat[3])
    matrix[2, 2] = 2 * (quat[3] ** 2 + quat[2] ** 2) - 1
    return matrix


def Rx(theta: units.Angle):
    """
    Creates a DCM representing a rotation about the x-axis.

    Args:
        theta (units.Angle): The rotation angle

    Returns:
        R: The resulting DCM.
    """
    from rsalib import Rotation

    return Rotation(
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )
    )


def Ry(theta: units.Angle):
    """
    Creates a DCM representing a rotation about the y-axis.

    Args:
        theta (units.Angle): The rotation angle

    Returns:
        R: The resulting DCM.
    """
    from rsalib import Rotation

    return Rotation(
        np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=float,
        )
    )


def Rz(theta: units.Angle):
    """
    Creates a DCM representing a rotation about the z-axis.

    Args:
        theta (units.Angle): The rotation angle

    Returns:
        R: The resulting DCM.
    """
    from rsalib import Rotation

    return Rotation(
        np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
    )


def _euler_angles_to_matrix(
    axis1: units.Angle, axis2: units.Angle, axis3: units.Angle, sequence: str = "ZYX"
) -> np.ndarray:
    """
    Creates a DCM from Euler angles.

    Args:
        a1 (units.Angle): The first Euler angle
        a2 (units.Angle): The second Euler angle
        a3 (units.Angle): The third Euler angle
        sequence (str): The rotation sequence. Defaults to "zyx".

    Returns:
        R: The resulting DCM.

    Raises:
        ValueError: If the sequence is invalid.
    """
    valid_sequences = {
        "zyz",
        "zyx",
        "zxy",
        "zxz",
        "yxz",
        "yxy",
        "yzx",
        "yzy",
        "xyz",
        "xyx",
        "xzy",
        "xzx",
    }
    sequence = sequence.lower()
    if sequence not in valid_sequences:
        raise ValueError("Invalid Euler angle sequence specified.")

    T_map = {"x": Rx, "y": Ry, "z": Rz}
    rotation = (
        T_map[sequence[2]](axis3)
        @ T_map[sequence[1]](axis2)
        @ T_map[sequence[0]](axis1)
    )

    return rotation.matrix


def _euler_angles_to_quat(
    axis1: units.Angle, axis2: units.Angle, axis3: units.Angle, sequence: str = "ZYX"
) -> np.ndarray:
    """
    Create a quaternion from Euler angles.

    Args:
        a1 (units.Angle): The first Euler angle
        a2 (units.Angle): The second Euler angle
        a3 (units.Angle): The third Euler angle
        sequence (str): The rotation sequence. Defaults to "zyx".

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """
    matrix = _euler_angles_to_matrix(axis1, axis2, axis3, sequence=sequence)
    quat = _matrix_to_quat(matrix)
    return quat


def _matrix_to_euler_angles(
    matrix: np.ndarray, sequence: str = "ZYX"
) -> Tuple[units.Angle, units.Angle, units.Angle]:
    """
    Convert a rotation matrix to Euler angles.

    Args:
        matrix (np.ndarray): 3x3 rotation matrix.
        sequence (str): The rotation sequence. Defaults to "zyx".

    Returns:
        Tuple[units.Angle, units.Angle, units.Angle]: The Euler angles (RADIANS).
    """
    valid_sequences = {
        "zyz",
        "zyx",
        "zxy",
        "zxz",
        "yxz",
        "yxy",
        "yzx",
        "yzy",
        "xyz",
        "xyx",
        "xzy",
        "xzx",
    }
    sequence = sequence.lower()
    if sequence not in valid_sequences:
        raise ValueError("Invalid Euler angle sequence specified.")

    # Inverse of _euler_angles_to_matrix
    if sequence == "zyx":
        sy = -matrix[0, 2]
        cy = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
        y = units.Angle.base_unit() * np.arctan2(sy, cy)

        sx = matrix[1, 2]
        cx = matrix[2, 2]
        x = units.Angle.base_unit() * np.arctan2(sx, cx)
        sz = matrix[0, 1]
        cz = matrix[0, 0]
        z = units.Angle.base_unit() * np.arctan2(sz, cz)

        return z, y, x
    else:
        raise NotImplementedError(
            "Euler angle extraction not implemented for this sequence."
        )


def _quat_to_euler_angles(
    quat: np.ndarray, sequence: str = "ZYX"
) -> Tuple[units.Angle, units.Angle, units.Angle]:
    """
    Convert a quaternion to Euler angles.

    Args:
        quat (np.ndarray): Quaternion as [x, y, z, w].
        sequence (str): The rotation sequence. Defaults to "zyx".

    Returns:
        Tuple[units.Angle, units.Angle, units.Angle]: The Euler angles (RADIANS).
    """
    matrix = _quat_to_matrix(quat)
    angles = _matrix_to_euler_angles(matrix, sequence=sequence)
    return angles


def _vectors_to_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Create a rotation matrix that rotates vector v1 to vector v2.

    Args:
        v1 (np.ndarray): Initial vector.
        v2 (np.ndarray): Target vector.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # If vectors are parallel, return identity
    if np.allclose(v1, v2):
        return np.eye(3)

    if np.allclose(v1, -v2):
        # Find a vector orthogonal to v1
        orth = (
            np.array([1, 0, 0])
            if not np.allclose(v1, [1, 0, 0])
            else np.array([0, 1, 0])
        )
        axis = np.cross(v1, orth)
        axis = axis / np.linalg.norm(axis)
        # Rodrigues formula for 180 deg rotation
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R_180 = np.eye(3) + 2 * K @ K  # since sin(pi)=0, (1-cos(pi))=2
        return R_180

    # Compute rotation axis (cross product)
    rotation_axis = np.cross(v1, v2)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Compute rotation angle
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)

    # Create skew-symmetric matrix
    K = skew_symmetric(rotation_axis)

    # Rodrigues rotation formula
    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return rotation_matrix


def _vectors_to_quat(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Create a quaternion that rotates vector v1 to vector v2.

    Args:
        v1 (np.ndarray): Initial vector.
        v2 (np.ndarray): Target vector.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """
    rotation_matrix = _vectors_to_matrix(v1, v2)
    quat = _matrix_to_quat(rotation_matrix)
    return quat


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix from an axis-angle representation.

    Args:
        axis (np.ndarray): Rotation axis (should be a unit vector).
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    matrix = np.array(
        [
            [c + ux**2 * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy**2 * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz**2 * C],
        ],
        dtype=float,
    )
    return matrix


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a quaternion from an axis-angle representation.

    Args:
        axis (np.ndarray): Rotation axis (should be a unit vector).
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """
    matrix = _axis_angle_to_matrix(axis, angle)
    quat = _matrix_to_quat(matrix)
    return quat


def _matrix_to_axis_angle(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert a rotation matrix to an axis-angle representation.

    Args:
        matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        Tuple[np.ndarray, float]: Rotation axis (unit vector) and rotation angle in radians.
    """
    angle = np.arccos((np.trace(matrix) - 1) / 2)
    if np.isclose(angle, 0):
        return np.array([1, 0, 0]), 0.0  # Arbitrary axis for zero rotation
    elif np.isclose(angle, np.pi):
        # Special case for 180 degree rotation
        R_plus_I = matrix + np.eye(3)
        axis = np.sqrt(np.diagonal(R_plus_I) / 2)
        axis = axis / np.linalg.norm(axis)
        return axis, angle
    else:
        rx = matrix[2, 1] - matrix[1, 2]
        ry = matrix[0, 2] - matrix[2, 0]
        rz = matrix[1, 0] - matrix[0, 1]
        axis = np.array([rx, ry, rz])
        axis = axis / (2 * np.sin(angle))
        axis = axis / np.linalg.norm(axis)
        return axis, angle


def _quat_to_axis_angle(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert a quaternion to an axis-angle representation.

    Args:
        quat (np.ndarray): Quaternion as [x, y, z, w].

    Returns:
        Tuple[np.ndarray, float]: Rotation axis (unit vector) and rotation angle in radians.
    """
    matrix = _quat_to_matrix(quat)
    axis, angle = _matrix_to_axis_angle(matrix)
    return axis, angle


def _mrp_to_quat(mrp: np.ndarray) -> np.ndarray:
    """
    Convert Modified Rodrigues Parameters (MRP) to a quaternion.

    Args:
        mrp (np.ndarray): Modified Rodrigues Parameters as [p1, p2, p3].

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """

    # Extract MRP components
    mrp1 = mrp[0]
    mrp2 = mrp[1]
    mrp3 = mrp[2]

    # MRP squared value
    mrp_sq = mrp1**2 + mrp2**2 + mrp3**2

    # Calculate quaternion components
    q1 = 2 * mrp1 / (1 + mrp_sq)
    q2 = 2 * mrp2 / (1 + mrp_sq)
    q3 = 2 * mrp3 / (1 + mrp_sq)
    qs = (1 - mrp_sq) / (1 + mrp_sq)

    # Assemble Quaternion
    quat = [q1, q2, q3, qs]

    # Return Quaternion
    return quat


def _quat_to_mrp(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to Modified Rodrigues Parameters (MRP).

    Args:
        quat (np.ndarray): Quaternion as [x, y, z, w].

    Returns:
        np.ndarray: Modified Rodrigues Parameters as [p1, p2, p3].
    """

    # Vector magnitude of the quaternion
    mag = np.linalg.norm(quat)

    # Check that the vector could reasonably be a quaternion
    if abs(1 - mag) >= 1e-8:
        raise ValueError(
            f"Quaternion magnitude is not close to 1. q = {quat}. Mag = {mag}"
        )

    # Normalize the quaternion
    quat = np.divide(quat, mag)

    # Store the components of the quaternion
    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]
    qs = quat[3]  # Scalar last definition used for quaternions

    # Compute in MRP format
    if qs != -1:  # check that it is not singular
        mrp1 = q1 / (1 + qs)
        mrp2 = q2 / (1 + qs)
        mrp3 = q3 / (1 + qs)

        mrp = [mrp1, mrp2, mrp3]

    else:  # if regular MRP singular, switch to shawdow set
        mrp1 = -q1 / (1 - qs)
        mrp2 = -q2 / (1 - qs)
        mrp3 = -q3 / (1 - qs)

        mrp = [mrp1, mrp2, mrp3]

    # If MRP describes a rotation greater than 180 deg, switch to the MRP shadow set
    # This is a Basilisk Standard
    if np.linalg.norm(mrp) > 1:
        mrp1 = -q1 / (1 - qs)
        mrp2 = -q2 / (1 - qs)
        mrp3 = -q3 / (1 - qs)

        mrp = [mrp1, mrp2, mrp3]

    # Return the MRP vector
    return mrp


def _rdt_to_matrix(ra: units.Angle, dec: units.Angle, twist: units.Angle) -> np.ndarray:
    """
    Create a rotation matrix from right ascension, declination, twist (RDT) parameters.

    Args:
        ra: Right ascension angle in radians.
        dec: Declination angle in radians.
        twist: Twist angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return constants.M_ICRF2IMAGE_ZERO_RDT @ (Rx(twist) @ Ry(-dec) @ Rz(ra)).matrix


def _rdt_to_quat(ra: units.Angle, dec: units.Angle, twist: units.Angle) -> np.ndarray:
    """
    Create a rotation from right ascension, declination, twist (RDT) parameters.
    Args:
        ra: Right ascension angle in radians.
        dec: Declination angle in radians.
        twist: Twist angle in radians.
    Returns:
        Rotation: Rotation object.
    """
    return _matrix_to_quat(_rdt_to_matrix(ra, dec, twist))


def _quat_to_rdt(quat: np.ndarray) -> Tuple[units.Angle, units.Angle, units.Angle]:
    """
    Converts a quaternion to right ascension, declination, and twist (RDT) angles.

    Parameters:
        quat (np.ndarray): A quaternion represented as a NumPy array of shape (4,) or (N, 4).

    Returns:
        Tuple[units.Angle, units.Angle, units.Angle]: A tuple containing the right ascension (ra), declination (dec), and twist angles.

    Notes:
        - The quaternion is first converted to a rotation matrix.
        - The resulting angles are computed as follows:
            - ra: Right ascension, computed from the rotation matrix.
            - dec: Declination, computed from the rotation matrix.
            - twist: Twist angle, computed from the rotation matrix.
        - Assumes the quaternion is in the format [w, x, y, z] or [x, y, z, w] depending on the implementation of _quat_to_matrix.
    """
    matrix = _quat_to_matrix(quat)
    matrix = constants.M_ICRF2IMAGE_ZERO_RDT.T @ matrix
    [ra, dec, twist] = _matrix_to_euler_angles(matrix, "ZYX")
    dec = -dec  # Negate for declination

    return (ra, dec, twist)
