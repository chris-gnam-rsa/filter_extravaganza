import numpy as np


class Rotation:
	"""Rotation represented as a 3x3 direction cosine matrix."""

	def __init__(self, matrix):
		self._matrix = self._validate_matrix(matrix)

	@property
	def matrix(self):
		return self._matrix.copy()

	@property
	def quaternion(self):
		m = self._matrix
		trace = m[0, 0] + m[1, 1] + m[2, 2]
		if trace > 0:
			s = 0.5 / np.sqrt(trace + 1.0)
			w = 0.25 / s
			x = (m[2, 1] - m[1, 2]) * s
			y = (m[0, 2] - m[2, 0]) * s
			z = (m[1, 0] - m[0, 1]) * s
		elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
			s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
			w = (m[2, 1] - m[1, 2]) / s
			x = 0.25 * s
			y = (m[0, 1] + m[1, 0]) / s
			z = (m[0, 2] + m[2, 0]) / s
		elif m[1, 1] > m[2, 2]:
			s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
			w = (m[0, 2] - m[2, 0]) / s
			x = (m[0, 1] + m[1, 0]) / s
			y = 0.25 * s
			z = (m[1, 2] + m[2, 1]) / s
		else:
			s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
			w = (m[1, 0] - m[0, 1]) / s
			x = (m[0, 2] + m[2, 0]) / s
			y = (m[1, 2] + m[2, 1]) / s
			z = 0.25 * s

		return np.array([x, y, z, w])

	@property
	def T(self):
		return Rotation(self._matrix.T)

	@staticmethod
	def _validate_matrix(matrix, tol=1e-6):
		mat = np.asarray(matrix, dtype=float)
		if mat.shape != (3, 3):
			raise ValueError("Rotation matrix must be 3x3.")
		identity = np.eye(3)
		should_be_identity = mat.T @ mat
		if not np.allclose(should_be_identity, identity, atol=tol):
			raise ValueError("Rotation matrix must be orthonormal.")
		det = np.linalg.det(mat)
		if not np.isclose(det, 1.0, atol=tol):
			raise ValueError("Rotation matrix must have det=1.")
		return mat

	@staticmethod
	def _validate_quaternion(quat, tol=1e-6):
		q = np.asarray(quat, dtype=float).reshape(4)
		norm = np.linalg.norm(q)
		if not np.isclose(norm, 1.0, atol=tol):
			raise ValueError("Quaternion must be unit length.")
		return q

	@staticmethod
	def _matrix_from_quaternion(quat):
		q = Rotation._validate_quaternion(quat)
		x, y, z, w = q

		xx = x * x
		yy = y * y
		zz = z * z
		xy = x * y
		xz = x * z
		yz = y * z
		wx = w * x
		wy = w * y
		wz = w * z

		return np.array(
			[
				[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
				[2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
				[2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
			],
			dtype=float,
		)

	@staticmethod
	def _axis_angle_matrix(axis, angle):
		c = np.cos(angle)
		s = np.sin(angle)
		if axis == "x":
			return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
		if axis == "y":
			return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
		if axis == "z":
			return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
		raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

	@classmethod
	def from_matrix(cls, matrix):
		return cls(matrix)

	@classmethod
	def from_quaternion(cls, quat, scalar_first=True):
		return cls(cls._matrix_from_quaternion(quat))

	@classmethod
	def from_euler(cls, angles, order="xyz", degrees=False):
		ang = np.asarray(angles, dtype=float).reshape(3)
		if len(order) != 3:
			raise ValueError("Euler order must have three axes.")
		order = order.lower()
		if set(order) - {"x", "y", "z"}:
			raise ValueError("Euler order must use axes 'x', 'y', 'z'.")
		if degrees:
			ang = np.deg2rad(ang)

		rot = np.eye(3)
		for axis, angle in zip(order, ang):
			rot = cls._axis_angle_matrix(axis, angle) @ rot
		return cls(rot)

	def rotate(self, vectors):
		vecs = np.asarray(vectors, dtype=float)
		if vecs.shape == (3,):
			return self._matrix @ vecs
		if vecs.ndim == 2 and vecs.shape[1] == 3:
			return (self._matrix @ vecs.T).T
		raise ValueError("Vectors must have shape (3,) or (N, 3).")

	def __matmul__(self, vectors):
		return self.rotate(vectors)

	def __mul__(self, vectors):
		return self.rotate(vectors)
