"""
Epoch: A comprehensive time representation class for astrodynamics.

This module provides the Epoch class for handling conversions between different
time scales and representations.

TIME SCALES vs REPRESENTATIONS
==============================

TIME SCALE: "Which clock are we reading?"
    Different physical clocks that tick at (nearly) the same rate but have
    offsets between them. The offset may be fixed or may change over time.

REPRESENTATION: "How do we write down the reading?"
    The format or number system used to express a time value. Any time scale
    can be expressed in any representation.

Time Scales
-----------
The following time scales are commonly used in astrodynamics:

UTC (Coordinated Universal Time)
    Civil time, kept synchronized with Earth's rotation via leap seconds.
    Discontinuous — jumps when leap seconds are inserted.
    Used for: TLEs, SGP4 propagation, civil timekeeping, timestamps.

TAI (International Atomic Time)
    Pure atomic clock time, ticks uniformly with SI seconds.
    TAI = UTC + ΔAT, where ΔAT is the cumulative leap second count.
    Currently (2024): TAI = UTC + 37 seconds.
    Used for: Fundamental time reference, basis for other scales.

TT (Terrestrial Time)
    Uniform time scale for Earth-surface astronomical calculations.
    TT = TAI + 32.184 seconds (fixed offset, by definition).
    Used for: Astronomical almanacs, Earth-based observations.

TDB (Barycentric Dynamical Time)
    Time scale for solar system barycentric ephemerides.
    TDB ≈ TT with periodic variations of ~1.7 ms amplitude (relativistic effects).
    For Earth-surface work, TDB ≈ TT to within ~2 ms.
    Used for: Planetary ephemerides, SPICE, interplanetary navigation.

GPS (GPS Time)
    Atomic time scale for GPS system.
    GPS = TAI - 19 seconds (fixed offset).
    GPS was synchronized with UTC on January 6, 1980 and has since diverged.
    Used for: GPS receivers, precise timing applications.

Relationships between time scales::

      GPS <-----------------  UTC <--- leap seconds --->  TAI --- +32.184s ---> TT ~ TDB
       |                       |                           |                       |
    TAI - 19s              (civil time)              (atomic time)          (dynamical time)


Representations
---------------
The following representations can express any time scale:

Julian Date (JD)
    Days (and fractions) since noon on January 1, 4713 BC (Julian proleptic calendar).
    A single floating-point number. Astronomers prefer it because one night
    stays within a single integer JD value.
    Notation: JD(UTC), JD(TT), JD(TDB), etc. to specify time scale.
    "JDE" or "JED" specifically means JD in TDB/TT scale.

Modified Julian Date (MJD)
    MJD = JD - 2400000.5
    Epoch is midnight on November 17, 1858. Smaller numbers, midnight epoch.
    Notation: MJD(UTC), MJD(TT), etc.

ISO 8601 / Calendar
    Year-month-day hour:minute:second format.
    Human-readable but complex for arithmetic.
    Example: "2024-07-01T12:01:00.000Z"

Unix Timestamp
    Seconds since midnight UTC on January 1, 1970.
    Conventionally in UTC scale (though leap second handling varies).

SPICE ET (Ephemeris Time)
    Seconds past J2000.0 (noon TT on January 1, 2000) in TDB scale.
    This is SPICE's native internal representation.

Seconds past J2000
    Like SPICE ET but can be in any time scale.
    J2000.0 = JD 2451545.0 = 2000-01-01T12:00:00 TT


AMBIGUITY WARNING
=================

When someone says "Julian Date" without qualification, the meaning depends on context:

    Community                    "JD" typically means
    ---------                    ---------------------
    Astronomers (ephemerides)    JD(TDB) or JD(TT)
    Satellite operators (TLEs)   JD(UTC)
    General software             JD(UTC)
    SPICE documentation          JD(TDB)

This class provides explicit methods to avoid ambiguity:
    - jd(scale) and mjd(scale) methods require explicit time scale
    - Factory methods require explicit scale specification

"""

import math
from datetime import datetime, timezone
from typing import Tuple, Union

import spiceypy as spice

from rsalib import units
from rsalib.utils.spice_utils import _load_lsk

# Valid time scales for validation
VALID_SCALES = frozenset({"UTC", "TAI", "TT", "TDB", "GPS"})


class Epoch:
    """
    Epoch class for precise time representation and conversion between time scales.

    This class stores time internally as SPICE Ephemeris Time (ET), which is
    seconds past J2000.0 in the TDB (Barycentric Dynamical Time) scale.

    All conversions to other time scales (UTC, TAI, TT, GPS) and representations
    (Julian Date, MJD, calendar, Unix timestamp) are computed on demand from this
    internal ET value.

    Attributes:
        _et (float): Internal time representation: seconds past J2000.0 in TDB scale.

    Examples:
        Creating epochs::

            >>> e = Epoch.from_utc("2024-07-01T12:00:00")
            >>> e = Epoch.from_jd(2460492.0, scale="UTC")
            >>> e = Epoch.from_jd(2460492.0, scale="TDB")  # Different instant!

        Getting time in different scales::

            >>> e = Epoch.from_utc("2024-07-01T12:00:00")
            >>> e.jd("UTC")   # Julian Date for SGP4
            >>> e.jd("TDB")   # Julian Date for SPICE
            >>> e.mjd("TAI")  # Modified Julian Date in TAI
    """

    #################
    ### Constants ###
    #################
    # Julian Date of J2000.0 epoch (2000-01-01T12:00:00 TT)
    J2000_JD: float = 2451545.0

    # Modified Julian Date offset: MJD = JD - MJD_OFFSET
    MJD_OFFSET: float = 2400000.5

    # Seconds per day
    SECONDS_PER_DAY: float = 86400.0

    # TT - TAI offset (fixed by definition)
    TT_TAI_OFFSET: float = 32.184

    # GPS - TAI offset (fixed by definition, GPS = TAI - 19)
    GPS_TAI_OFFSET: float = -19.0

    # Unix epoch as Julian Date (UTC): 1970-01-01T00:00:00 UTC
    UNIX_EPOCH_JD: float = 2440587.5

    ####################
    ### Construction ###
    ####################
    def __init__(self, input: Union[float, datetime]):
        """
        Initialize an Epoch from a SPICE Ephemeris Time (ET) value or a datetime.

        Args:
            input: Either a float (seconds past J2000.0 in TDB scale) or a datetime object.
        """
        _load_lsk()
        if isinstance(input, (float, int)):
            self._et = float(input)
        elif isinstance(input, datetime):
            self._et = Epoch.from_datetime(input)._et
        else:
            raise TypeError("Epoch input must be a float (ET) or a datetime object.")

    ############################
    ### Factory methods: UTC ###
    ############################
    @classmethod
    def from_utc(cls, utc_string: str) -> "Epoch":
        """
        Create an Epoch from a UTC time string.

        Args:
            utc_string: UTC time string in ISO 8601 format or any format recognized by SPICE.

        Returns:
            Epoch representing the specified UTC time.
        """
        _load_lsk()
        et = spice.str2et(utc_string)
        return cls(et)

    @classmethod
    def from_datetime(cls, dt: datetime) -> "Epoch":
        """
        Create an Epoch from a Python datetime object.

        Args:
            dt: Python datetime object. If naive (no timezone), it is assumed to be UTC.

        Returns:
            Epoch representing the specified datetime.
        """
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt.astimezone(timezone.utc)
        utc_string = dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")
        return cls.from_utc(utc_string)

    @classmethod
    def from_unix(cls, timestamp: float) -> "Epoch":
        """
        Create an Epoch from a Unix timestamp.

        Args:
            timestamp: Seconds since 1970-01-01T00:00:00 UTC (Unix epoch).

        Returns:
            Epoch representing the specified Unix time.
        """
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return cls.from_datetime(dt)

    ####################################
    ### Factory methods: Julian Date ###
    ####################################
    @classmethod
    def from_jd(cls, jd: Union[float, Tuple[float, float]], scale: str) -> "Epoch":
        """
        Create an Epoch from a Julian Date with explicit time scale.

        Args:
            jd: Julian Date as a single value or as (integer_part, fractional_part)
                for higher precision.
            scale: Time scale of the input. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Epoch representing the specified Julian Date.

        Notes:
            The same Julian Date value represents DIFFERENT instants in different
            time scales. JD 2460492.5 in UTC is about 69 seconds earlier than
            JD 2460492.5 in TDB.
        """
        _load_lsk()

        # Handle tuple input for high precision
        if isinstance(jd, tuple):
            jd_val = jd[0] + jd[1]
        else:
            jd_val = float(jd)

        scale = scale.upper()
        if scale not in VALID_SCALES:
            raise ValueError(
                f"Unrecognized time scale: '{scale}'. "
                f"Must be one of: {', '.join(sorted(VALID_SCALES))}"
            )

        if scale == "TDB":
            et = spice.unitim(jd_val, "JED", "ET")

        elif scale == "TT":
            # Convert JD(TT) to ET via SPICE
            et = spice.unitim(jd_val, "JDTDT", "ET")

        elif scale == "TAI":
            # TAI = TT - 32.184s, so JD(TAI) = JD(TT) - 32.184/86400
            jd_tt = jd_val + cls.TT_TAI_OFFSET / cls.SECONDS_PER_DAY
            return cls.from_jd(jd_tt, scale="TT")

        elif scale == "GPS":
            # GPS = TAI - 19s, so JD(GPS) = JD(TAI) - 19/86400
            jd_tai = jd_val - cls.GPS_TAI_OFFSET / cls.SECONDS_PER_DAY
            return cls.from_jd(jd_tai, scale="TAI")

        elif scale == "UTC":
            et = cls._jd_utc_to_et(jd_val)

        return cls(et)

    @classmethod
    def from_mjd(cls, mjd: float, scale: str) -> "Epoch":
        """
        Create an Epoch from a Modified Julian Date with explicit time scale.

        Args:
            mjd: Modified Julian Date (MJD = JD - 2400000.5).
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Epoch representing the specified MJD.
        """
        jd = mjd + cls.MJD_OFFSET
        return cls.from_jd(jd, scale=scale)

    ###########################################
    ### Factory methods: Seconds past J2000 ###
    ###########################################
    @classmethod
    def from_et(cls, et: float) -> "Epoch":
        """
        Create an Epoch from SPICE Ephemeris Time (ET).

        ET is seconds past J2000.0 (2000-01-01T12:00:00 TT) in the TDB scale.
        This is SPICE's native time representation.

        Args:
            et: Ephemeris Time in seconds past J2000.0 (TDB).

        Returns:
            Epoch representing the specified ET.
        """
        return cls(et)

    @classmethod
    def from_seconds_past_j2000(cls, seconds: float, scale: str) -> "Epoch":
        """
        Create an Epoch from seconds past J2000.0 in a specified time scale.

        Args:
            seconds: Seconds past J2000.0 (2000-01-01T12:00:00 in the specified scale).
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Epoch representing the specified time.
        """
        jd = cls.J2000_JD + seconds / cls.SECONDS_PER_DAY
        return cls.from_jd(jd, scale=scale)

    ###################################
    ### Output: Common properties   ###
    ###################################
    @property
    def et(self) -> float:
        """
        SPICE Ephemeris Time: seconds past J2000.0 in TDB scale.

        This is the internal representation of the Epoch.

        Returns:
            Seconds past J2000.0 (TDB).
        """
        return self._et

    @property
    def utc(self) -> str:
        """
        UTC time as ISO 8601 string.

        Returns:
            UTC string in format "YYYY-MM-DDTHH:MM:SS.ssssss".
        """
        return spice.et2utc(self._et, "ISOC", 6)

    @property
    def datetime(self) -> datetime:
        """
        UTC time as timezone-aware Python datetime object.

        Returns:
            Datetime object with UTC timezone.
        """
        utc_str = spice.et2utc(self._et, "ISOC", 6)
        return datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%f").replace(
            tzinfo=timezone.utc
        )

    @property
    def unix(self) -> float:
        """
        Unix timestamp: seconds since 1970-01-01T00:00:00 UTC.

        Returns:
            Unix timestamp.
        """
        return self.datetime.timestamp()

    ##################################################
    ### Output: Seconds past J2000 (by time scale) ###
    ##################################################
    def seconds_past_j2000(self, scale: str) -> float:
        """
        Get seconds past J2000.0 in a specified time scale.

        Args:
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Seconds past J2000.0 in the specified time scale.
        """
        scale = scale.upper()
        if scale not in VALID_SCALES:
            raise ValueError(
                f"Unrecognized time scale: '{scale}'. "
                f"Must be one of: {', '.join(sorted(VALID_SCALES))}"
            )

        if scale == "TDB":
            return self._et
        elif scale == "TT":
            return spice.unitim(self._et, "ET", "TT")
        elif scale == "TAI":
            return spice.unitim(self._et, "ET", "TT") - self.TT_TAI_OFFSET
        elif scale == "GPS":
            tai = spice.unitim(self._et, "ET", "TT") - self.TT_TAI_OFFSET
            return tai + self.GPS_TAI_OFFSET
        elif scale == "UTC":
            return (self.jd("UTC") - self.J2000_JD) * self.SECONDS_PER_DAY

    ###########################################
    ### Output: Julian Date (by time scale) ###
    ###########################################
    def to_jd(self, scale: str) -> float:
        """
        Get Julian Date in a specified time scale.

        Args:
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Julian Date in the specified time scale.
        """
        scale = scale.upper()
        if scale not in VALID_SCALES:
            raise ValueError(
                f"Unrecognized time scale: '{scale}'. "
                f"Must be one of: {', '.join(sorted(VALID_SCALES))}"
            )

        if scale == "TDB":
            return spice.unitim(self._et, "ET", "JED")
        elif scale == "TT":
            return spice.unitim(self._et, "ET", "JDTDT")
        elif scale == "TAI":
            return self.to_jd("TT") - self.TT_TAI_OFFSET / self.SECONDS_PER_DAY
        elif scale == "GPS":
            return self.to_jd("TAI") - self.GPS_TAI_OFFSET / self.SECONDS_PER_DAY
        elif scale == "UTC":
            return self._datetime_to_jd_utc(self.datetime)

    def jd_utc_split(self) -> Tuple[float, float]:
        """
        Get Julian Date as a two-part tuple for higher precision.

        For SGP4 compatibility (scale="UTC"), returns:
        - First element: JD at midnight of the date (ends in .5)
        - Second element: Fractional day offset for time of day

        Args:
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Tuple of (JD_base, JD_fraction) where sum equals full JD.
        """
        jd_val = self.to_jd("UTC")

        # SGP4-compatible split: midnight + fraction of day
        jd_midnight = math.floor(jd_val - 0.5) + 0.5
        fr = jd_val - jd_midnight
        return jd_midnight, fr

    ####################################################
    ### Output: Modified Julian Date (by time scale) ###
    ####################################################
    def to_mjd(self, scale: str) -> float:
        """
        Get Modified Julian Date in a specified time scale.

        MJD = JD - 2400000.5

        Args:
            scale: Time scale. Must be one of: "UTC", "TAI", "TT", "TDB", "GPS".

        Returns:
            Modified Julian Date in the specified time scale.
        """
        return self.to_jd(scale) - self.MJD_OFFSET

    #############################
    ### Output: GPS-specific  ###
    #############################
    @property
    def gps_week_and_seconds(self) -> Tuple[int, float]:
        """
        GPS week number and seconds into the week.

        GPS week 0 started on January 6, 1980 00:00:00 UTC.

        Returns:
            Tuple of (GPS_week, seconds_of_week).
        """
        # GPS epoch: January 6, 1980 00:00:00 UTC
        gps_epoch = Epoch.from_utc("1980-01-06T00:00:00")

        # Use the JD difference in TAI scale (TAI is uniform, no leap seconds)
        delta_days = self.jd("TAI") - gps_epoch.jd("TAI")
        delta_seconds = delta_days * self.SECONDS_PER_DAY

        seconds_per_week = 7 * self.SECONDS_PER_DAY
        week = int(delta_seconds // seconds_per_week)
        seconds_of_week = delta_seconds % seconds_per_week

        return week, seconds_of_week

    #################################
    ### Time scale offset queries ###
    #################################
    @property
    def delta_at(self) -> float:
        """
        ΔAT = TAI - UTC at this epoch (cumulative leap seconds).

        Returns:
            TAI - UTC offset in seconds.
        """
        return spice.deltet(self._et, "UTC") - self.TT_TAI_OFFSET

    @property
    def delta_t(self) -> float:
        """
        ΔT = TT - UTC at this epoch.

        This is the sum of the TAI-UTC offset (leap seconds) and the
        fixed TT-TAI offset (32.184 seconds).

        Returns:
            TT - UTC offset in seconds.
        """
        return self.delta_at + self.TT_TAI_OFFSET

    ##############################
    ### Private helper methods ###
    ##############################
    @staticmethod
    def _datetime_to_jd_utc(dt: datetime) -> float:
        """
        Convert datetime to Julian Date in UTC scale using Vallado's algorithm.

        This computes JD directly from calendar components, matching the
        traditional jday() function used with SGP4.
        """
        year = dt.year
        mon = dt.month
        day = dt.day
        hr = dt.hour
        minute = dt.minute
        sec = dt.second + dt.microsecond / 1e6

        jd = (
            367.0 * year
            - int(7 * (year + int((mon + 9) / 12.0)) * 0.25)
            + int(275 * mon / 9.0)
            + day
            + 1721013.5
        )
        fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0

        return jd + fr

    @classmethod
    def _jd_utc_to_et(cls, jd_utc: float) -> float:
        """
        Convert Julian Date (UTC) to SPICE ET via calendar decomposition.

        This ensures proper leap second handling by going through SPICE's
        UTC string parser.
        """
        # Convert JD to calendar date using standard algorithm
        z = int(jd_utc + 0.5)
        f = (jd_utc + 0.5) - z

        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)

        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)

        day = b - d - int(30.6001 * e)
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715

        # Convert fractional day to time
        day_frac = f
        hours = int(day_frac * 24)
        day_frac = day_frac * 24 - hours
        minutes = int(day_frac * 60)
        day_frac = day_frac * 60 - minutes
        seconds = day_frac * 60

        # Format and convert via SPICE
        utc_str = f"{year:04d}-{month:02d}-{day:02d}T{hours:02d}:{minutes:02d}:{seconds:012.9f}"
        _load_lsk()
        return spice.str2et(utc_str)

    ############################
    ### Comparison operators ###
    ############################
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Epoch):
            return self._et == other._et
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Epoch):
            return self._et != other._et
        return NotImplemented

    def __lt__(self, other: "Epoch") -> bool:
        if isinstance(other, Epoch):
            return self._et < other._et
        return NotImplemented

    def __le__(self, other: "Epoch") -> bool:
        if isinstance(other, Epoch):
            return self._et <= other._et
        return NotImplemented

    def __gt__(self, other: "Epoch") -> bool:
        if isinstance(other, Epoch):
            return self._et > other._et
        return NotImplemented

    def __ge__(self, other: "Epoch") -> bool:
        if isinstance(other, Epoch):
            return self._et >= other._et
        return NotImplemented

    ##################
    ### Arithmetic ###
    ##################
    def __add__(self, time_delta: units.Time) -> "Epoch":
        """
        Add a time delta to this epoch, returning a new Epoch.

        Args:
            time_delta: Time delta to add.

        Returns:
            New Epoch offset by the specified time.
        """
        if isinstance(time_delta, units.Time):
            return Epoch(self._et + time_delta.to_float("second"))
        return NotImplemented

    def __radd__(self, time_delta: units.Time) -> "Epoch":
        """Add time delta to this epoch (reverse operand)."""
        return self.__add__(time_delta)

    def __sub__(self, other: Union["Epoch", units.Time]) -> Union["Epoch", units.Time]:
        """
        Subtract a time delta or another Epoch from this epoch.

        Args:
            other: If Epoch, returns the difference as units.Time.
                   If units.Time, returns a new Epoch offset by -time_delta.

        Returns:
            If subtracting units.Time: new Epoch.
            If subtracting Epoch: difference as units.Time.
        """
        if isinstance(other, Epoch):
            return units.Time.base_unit(self._et - other._et)
        elif isinstance(other, units.Time):
            return Epoch(self._et - other.to_float("second"))
        return NotImplemented

    ##############################
    ### String representations ###
    ##############################
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Epoch(et={self._et}, utc='{self.utc}')"

    def __str__(self) -> str:
        """Human-readable UTC string."""
        return self.utc

    def __format__(self, format_spec: str) -> str:
        """
        Format the epoch using format specifiers.

        Format specifiers:
            'utc' or '': UTC ISO string
            'et': Ephemeris Time
            'unix': Unix timestamp
        """
        if format_spec in ("", "utc"):
            return self.utc
        elif format_spec == "et":
            return str(self._et)
        elif format_spec == "unix":
            return str(self.unix)
        else:
            raise ValueError(f"Unknown format specifier: {format_spec}")
