import math
import numbers
from abc import ABC
from typing import Union


class Unit(ABC):
    _unit_to_factor = None  # subclasses must override

    @classmethod
    def base_unit(cls):
        """
        Returns an instance of the class with value 1 and the unit whose scale is 1.0.
        """
        if not hasattr(cls, "_unit_to_factor") or cls._unit_to_factor is None:
            raise NotImplementedError(
                f"You must define _unit_to_factor on {cls.__name__}"
            )

        for unit, factor in cls._unit_to_factor.items():
            if factor == 1.0:
                return cls(1, unit)

        raise ValueError(f"No base unit (scale 1.0) defined for {cls.__name__}")

    def __init__(self, value, unit):
        # Check for valid mapping in subclass
        if not hasattr(self, "_unit_to_factor") or self._unit_to_factor is None:
            raise NotImplementedError(
                f"You must define _unit_to_factor on {self.__class__.__name__}"
            )
        if unit not in self._unit_to_factor:
            raise ValueError(f"Unrecognized unit for {self.__class__.__name__}: {unit}")
        self._unit = unit
        self._value = value
        self._base_value = value * self._unit_to_factor[unit]

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self.__class__(self._value * other, unit=self._unit)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # Convert other to this unit
        other_in_self_unit = other.to_float(self._unit)
        return self.__class__(self._value + other_in_self_unit, unit=self._unit)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # Convert other to this unit
        other_in_self_unit = other.to_float(self._unit)
        return self.__class__(self._value - other_in_self_unit, unit=self._unit)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __float__(self):
        raise TypeError(
            f"Automatic casting to float is not allowed for {self.__class__.__name__}. Use .to_float() instead."
        )

    def to_float(self, unit: Union[str, "Unit"]) -> float:
        scale = 1.0
        if isinstance(unit, str):
            if unit not in self._unit_to_factor:
                raise ValueError(
                    f"Unrecognized unit for {self.__class__.__name__}: {unit}"
                )
            scale = self._unit_to_factor[unit]
        elif isinstance(unit, Unit):
            scale = self._unit_to_factor[unit._unit]

        return float(self._base_value / scale)

    def __str__(self):
        return f"{self.__class__.__name__}({self._value}, unit='{self._unit}')"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value}, unit='{self._unit}')"

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        # Convert to the same scale and compare values:
        return math.isclose(self.to_float(self._unit), other.to_float(self._unit))

    def __neg__(self):
        return self.__class__(-self._value, unit=self._unit)


class Angle(Unit):
    _unit_to_factor = {
        "radian": 1.0,
        "degree": math.pi / 180,
        "arcmin": math.pi / (180 * 60),
        "arcsec": math.pi / (180 * 3600),
    }

    def __truediv__(self, other):
        from .units import AngularVelocity, Time

        if isinstance(other, Time):
            # Calculate base values (radians and seconds)
            value_in_rad = self.to_float("radian")
            value_in_s = other.to_float("second")
            result = value_in_rad / value_in_s
            return result * AngularVelocity.base_unit()
        return NotImplemented

    def __float__(self):
        # Always return value in radians (base units)
        return float(self._base_value)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert all Angle instances to float (radians) before applying the ufunc
        float_inputs = [float(x) if isinstance(x, Angle) else x for x in inputs]
        return ufunc(*float_inputs, **kwargs)


DEGREES = Angle(1.0, "degree")
DEG = DEGREES
RADIANS = Angle(1.0, "radian")
RAD = RADIANS
ARCMINUTES = Angle(1.0, "arcmin")
ARCMIN = ARCMINUTES
ARCSECONDS = Angle(1.0, "arcsec")
ARCSEC = ARCSECONDS


class Distance(Unit):
    _unit_to_factor = {"mm": 0.001, "meter": 1.0, "km": 1000.0, "AU": 1.496e11}

    def __truediv__(self, other):
        from .units import Time, Velocity

        if isinstance(other, Time):
            # Calculate base values (meters and seconds)
            value_in_m = self.to_float("meter")
            value_in_s = other.to_float("second")
            result = value_in_m / value_in_s
            return result * Velocity.base_unit()
        return NotImplemented


MILLIMETER = Distance(1.0, "mm")
MM = MILLIMETER
METER = Distance(1.0, "meter")
M = METER
KILOMETER = Distance(1.0, "km")
KM = KILOMETER
ASTRONOMICAL_UNIT = Distance(1.0, "AU")
AU = ASTRONOMICAL_UNIT


class Time(Unit):
    _unit_to_factor = {
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
        "day": 86400.0,
    }


SECONDS = Time(1.0, "second")
MINUTES = Time(1.0, "minute")
HOURS = Time(1.0, "hour")
DAYS = Time(1.0, "day")


class Velocity(Unit):
    _unit_to_factor = {
        "m/s": 1.0,
        "km/s": 1000.0,
        "km/h": 1000.0 / 3600.0,
    }

    def __truemul__(self, other):
        from .units import Time

        if isinstance(other, Time):
            # Calculate base values (meters and seconds)
            value_in_mps = self.to_float("m/s")
            value_in_s = other.to_float("second")
            result = value_in_mps * value_in_s
            return result * Distance.base_unit()
        return NotImplemented


METERS_PER_SECOND = Velocity(1.0, "m/s")
MPS = METERS_PER_SECOND
M_PER_S = MPS
KILOMETERS_PER_SECOND = Velocity(1.0, "km/s")
KPS = KILOMETERS_PER_SECOND
KM_PER_S = KPS
KILOMETERS_PER_HOUR = Velocity(1.0, "km/h")
KPH = KILOMETERS_PER_HOUR
KM_PER_H = KPH


class AngularVelocity(Unit):
    _unit_to_factor = {
        "rad/s": 1.0,
        "deg/s": math.pi / 180,
    }


RADIANS_PER_SECOND = AngularVelocity(1.0, "rad/s")
RADPS = RADIANS_PER_SECOND
RAD_PER_S = RADPS
DEGREES_PER_SECOND = AngularVelocity(1.0, "deg/s")
DEGPS = DEGREES_PER_SECOND
DEG_PER_S = DEGPS
