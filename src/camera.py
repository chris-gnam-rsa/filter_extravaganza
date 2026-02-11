import numpy as np


class Camera:
	"""Pinhole camera with configurable pose and intrinsics."""

	def __init__(
		self,
		position,
		orientation,
		focal_length,
		sensor_size,
		resolution,
	):
		self.position = np.asarray(position, dtype=float).reshape(3)
		self.orientation = orientation
		self.focal_length = float(focal_length)
		self.sensor_size = np.asarray(sensor_size, dtype=float).reshape(2)
		self.resolution = np.asarray(resolution, dtype=int).reshape(2)

	@property
	def _orientation_matrix(self):
		if hasattr(self.orientation, "matrix"):
			return self.orientation.matrix
		return np.asarray(self.orientation, dtype=float).reshape(3, 3)

	@property
	def _world_to_camera(self):
		return self._orientation_matrix.T

	def project_points(self, points_world):
		"""Project 3D world points into pixel coordinates.

		Returns pixels in (u, v) with origin at top-left.
		"""
		points = np.asarray(points_world, dtype=float)
		if points.ndim == 1:
			points = points.reshape(1, 3)

		rel = points - self.position
		cam = (self._world_to_camera @ rel.T).T

		z = cam[:, 2]
		valid = z > 0
		x_img = np.full_like(z, np.nan, dtype=float)
		y_img = np.full_like(z, np.nan, dtype=float)
		x_img[valid] = self.focal_length * cam[valid, 0] / z[valid]
		y_img[valid] = self.focal_length * cam[valid, 1] / z[valid]

		sensor_w, sensor_h = self.sensor_size
		res_w, res_h = self.resolution
		u = (x_img / sensor_w + 0.5) * res_w
		v = (0.5 - y_img / sensor_h) * res_h
		pixels = np.stack([u, v], axis=1)
		in_bounds = (
			(u >= 0)
			& (u < res_w)
			& (v >= 0)
			& (v < res_h)
			& valid
		)
		pixels = pixels[in_bounds]

		return pixels, in_bounds

	def project_directions(self, directions_world):
		"""Project 3D world directions into pixel coordinates."""
		directions = np.asarray(directions_world, dtype=float)
		if directions.ndim == 1:
			directions = directions.reshape(1, 3)

		rel = directions
		cam = (self._world_to_camera @ rel.T).T

		z = cam[:, 2]
		valid = z > 0
		x_img = np.full_like(z, np.nan, dtype=float)
		y_img = np.full_like(z, np.nan, dtype=float)
		x_img[valid] = self.focal_length * cam[valid, 0] / z[valid]
		y_img[valid] = self.focal_length * cam[valid, 1] / z[valid]

		sensor_w, sensor_h = self.sensor_size
		res_w, res_h = self.resolution
		u = (x_img / sensor_w + 0.5) * res_w
		v = (0.5 - y_img / sensor_h) * res_h
		pixels = np.stack([u, v], axis=1)
		in_bounds = (
			(u >= 0)
			& (u < res_w)
			& (v >= 0)
			& (v < res_h)
			& valid
		)
		pixels = pixels[in_bounds]

		return pixels, in_bounds

	def pixel_to_rays(self, pixels):
		"""Cast pixel coordinates into unit rays in world coordinates."""
		dirs_cam = self.pixel_to_rays_body(pixels)

		dirs_world = (self._orientation_matrix @ dirs_cam.T).T
		return dirs_world

	def pixel_to_rays_body(self, pixels):
		"""Cast pixel coordinates into unit rays in body coordinates."""
		pix = np.asarray(pixels, dtype=float)
		if pix.ndim == 1:
			pix = pix.reshape(1, 2)

		res_w, res_h = self.resolution
		sensor_w, sensor_h = self.sensor_size
		x = (pix[:, 0] / res_w - 0.5) * sensor_w
		y = (0.5 - pix[:, 1] / res_h) * sensor_h

		dirs_cam = np.stack([x, y, np.full_like(x, self.focal_length)], axis=1)
		norms = np.linalg.norm(dirs_cam, axis=1, keepdims=True)
		dirs_cam = dirs_cam / norms
		return dirs_cam
