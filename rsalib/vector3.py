from abc import ABC

import numpy as np

from rsalib import units


class Vector3(ABC):
    _unit_class = units.Unit

    def __init__(self, x, y=None, z=None, units=None):
        self._unit = self._unit_class.base_unit()

        # Accept a single np.ndarray and a unit
        if isinstance(x, np.ndarray) and x.size == 3 and units is not None:
            x = x.reshape(3)
            if not isinstance(units, self._unit_class):
                raise TypeError(
                    f"units parameter must be a {self._unit_class.__name__} instance"
                )
            self._v = x.astype(float) * units.to_float(self._unit_class.base_unit())
            return

        # If a unit is provided, x, y, and z must be numbers (not instances of _unit_class)
        if units is not None:
            if not isinstance(units, self._unit_class):
                raise TypeError(
                    f"units parameter must be a {self._unit_class.__name__} instance"
                )
            if any(isinstance(val, self._unit_class) for val in (x, y, z)):
                raise TypeError("x, y, and z must be numbers when a unit is provided")
            self._v = np.array([x, y, z], dtype=float) * units.to_float(
                self._unit_class.base_unit()
            )
            return

        # All arguments are likely units.Unit instances (x, y, z themselves have unit attribute)
        if all(isinstance(u, self._unit_class) for u in (x, y, z)):
            self._v = np.array(
                [
                    x.to_float(self._unit_class.base_unit()),
                    y.to_float(self._unit_class.base_unit()),
                    z.to_float(self._unit_class.base_unit()),
                ],
                dtype=float,
            )
            return

        raise ValueError(
            f"Vector3 expects either 3 {self._unit_class.__name__} objects, "
            f"3 numbers and a {self._unit_class.__name__} as the 4th arg, "
            f"or a numpy array and a {self._unit_class.__name__} as the 2nd arg"
        )

    def __str__(self):
        return (
            f"Vector3({self._v[0]}, {self._v[1]}, {self._v[2]}, units='{self._unit}')"
        )

    def __repr__(self):
        return self.__str__()

    def to(self, unit) -> np.ndarray:
        if isinstance(unit, str):
            u = self._unit_class(1, unit)

        elif isinstance(unit, self._unit_class):
            u = unit

        else:
            raise TypeError("unit must be a string or a matching units.Unit instance")

        factor = float(u)
        return np.array([self._v[0] / factor, self._v[1] / factor, self._v[2] / factor])

    @property
    def v(self):
        return self._v

    @property
    def values(self):
        return self._v

    @property
    def x(self):
        return self._v[0] * self._unit_class.base_unit()

    @property
    def y(self):
        return self._v[1] * self._unit_class.base_unit()

    @property
    def z(self):
        return self._v[2] * self._unit_class.base_unit()

    def __eq__(self, other):
        if self._unit_class != other._unit_class:
            return False

        if isinstance(other, Vector3):
            return np.allclose(self._v, other._v)

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, index):
        """
        Allows indexing to access vector components.

        Args:
            index (int): Index of the component (0 for x, 1 for y, 2 for z).

        Returns:
            float: The value of the specified component.
        """
        if index < 0 or index > 2:
            raise IndexError("Index must be 0, 1, or 2.")

        return self._v[index] * self._unit_class.base_unit()

    def __array__(self):
        """
        Converts the Vector3 object to a numpy array when used inside np.array().

        Returns:
            np.ndarray: The vector as a numpy array.
        """
        return self._v

    def __add__(self, other):
        """
        Adds two vectors component-wise.

        Args:
            other (Vector3): The vector to add.

        Returns:
            Vector3: The resulting vector.

        Raises:
            TypeError: If the other object is not a Vector3.
        """
        if self._unit_class != other._unit_class:
            raise TypeError(
                "Addition is only supported between Vector3 objects of the same unit type."
            )

        if not isinstance(other, Vector3):
            raise TypeError("Addition is only supported between Vector3 objects.")

        return Vector3(self._v + other._v)

    def __sub__(self, other):
        """
        Subtracts two vectors component-wise.

        Args:
            other (Vector3): The vector to subtract.

        Returns:
            Vector3: The resulting vector.

        Raises:
            TypeError: If the other object is not a Vector3.
        """
        if self._unit_class != other._unit_class:
            raise TypeError(
                "Subtraction is only supported between Vector3 objects of the same unit type."
            )

        if not isinstance(other, Vector3):
            raise TypeError("Subtraction is only supported between Vector3 objects.")

        return Vector3(self.v - other.v)

    def mag(self):
        """
        Returns the magnitude (norm) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        return np.linalg.norm(self._v)

    def direction(self):
        """
        Returns the direction (unit vector) of the vector.

        Returns:
            Vector3: The unit vector in the direction of the original vector.

        Raises:
            ValueError: If the vector magnitude is zero.
        """
        mag = self.mag()
        if mag == 0:
            raise ValueError("Cannot determine direction of a zero vector.")
        return Vector3(self._v / mag)


# Generalized array container for Vector3-like objects
class Vector3Array:
    def __init__(self, data, unit=None, frame=None):
        """
        data: (N, 3) array-like or list of Vector3
        unit: units.Unit instance (optional, overrides Vector3 units)
        frame: ReferenceFrame (optional)
        """
        if (
            isinstance(data, (list, tuple))
            and len(data) > 0
            and isinstance(data[0], Vector3)
        ):
            if unit is not None:
                raise ValueError(
                    "Cannot specify unit when constructing from Vector3 list"
                )
            # Construct from list of Vector3
            unit_class = data[0]._unit_class
            if not all(getattr(v, "_unit_class", None) == unit_class for v in data):
                raise TypeError("All Vector3 elements must have the same _unit_class")
            arr = np.stack([v.v for v in data])
            self._unit_class = unit_class
            self._frame = getattr(data[0], "frame", frame)
        else:
            arr = np.array(data, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("data must be (N, 3) array or list of Vector3")
            self._unit_class = getattr(self, "_unit_class", units.Unit)
            self._frame = frame
        self._data = arr
        self._unit = unit if unit is not None else self._unit_class.base_unit()

    @property
    def data(self):
        return self._data

    @property
    def frame(self):
        return self._frame

    def to(self, unit):
        """Convert to a different unit."""
        if isinstance(unit, str):
            u = self._unit_class(1, unit)
        elif isinstance(unit, self._unit_class):
            u = unit
        else:
            raise TypeError("unit must be a string or a matching units.Unit instance")
        factor = float(u)
        return self._data / factor

    def mag(self):
        """Return magnitudes of all vectors."""
        return np.linalg.norm(self._data, axis=1)

    def direction(self):
        """Return unit vectors for all vectors."""
        mags = self.mag()[:, None]
        if np.any(mags == 0):
            raise ValueError("Zero vector in array; cannot normalize.")
        return self._data / mags

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        # Return a Vector3 of the right type
        v = self._data[idx]
        return self._unit_class(v[0], v[1], v[2], unit=self._unit)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._data.shape}, unit={self._unit}, frame={self._frame})"

    def __repr__(self) -> str:
        return self.__str__()
