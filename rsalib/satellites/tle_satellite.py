import pickle
import re
from pathlib import Path

import numpy as np
from sgp4 import api, omm

from rsalib import Epoch
from rsalib.frames import TEME, ReferenceFrame, SpiceFrame
from rsalib.state_vector import Position, Velocity


class TLESatellite:
    """TLESatellite class using SGP4 propagation for TLE-defined objects (output in TEME frame)."""

    def __init__(
        self,
        name: str,
        cat_id: int,
    ) -> None:
        """TLESatellite constructor.

        Args:
            name (str): name of TLE satellite.
            cat_id (int): NORAD satellite catalog ID.
        """
        self._name = name
        self._cat_id = cat_id


class TLESatelliteArray:
    def __init__(
        self,
        satellites: list,
        names: list,
        cat_ids: list,
    ) -> None:
        """TLESatelliteArray constructor for TLE-defined satellites.

        Args:
            satellites (list): List of TLESatellite objects
            names (list): List of corresponding TLE satellite names
            cat_ids (list): List of corresponding NORAD satellite catalog IDs.
        """
        # Set private properties
        self._current = 0
        self._size = len(satellites)
        self._names = names
        self._cat_ids = cat_ids

        # Initialize sgp4 satellite objects
        self._satrec_array = api.SatrecArray(satellites)

        self._teme_frame = TEME()

    def __getitem__(self, i) -> TLESatellite:
        this_target = TLESatellite(
            name=self.name(i),
            cat_id=self.cat_id(i),
        )
        return this_target

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self) -> TLESatellite:
        if self._current < self.size:
            this_target = self.__getitem__(self._current)
            self._current += 1
            return this_target
        raise StopIteration

    def __len__(self) -> int:
        return self._size

    @classmethod
    def from_file(
        cls,
        filepath: Path = Path.home() / "dat" / "spacetrack_catalog.pkl",
        filter_str: str = None,
        exclude_small_debris: bool = True,
    ):
        """Generate TLESatelliteArray object from file containing TLE-defined satellites.

        Args:
            filepath (Path): Path to file containing catalog from Space-Track.org. (csv, xml, pkl)
            filter_str (str, optional): Regular expression string that can be used to filter objects kept in the catalog. Defaults to None.
            exclude_small_debris (bool, optional): If True, exclude debris objects, which are typically too small to be detected and make up >50% of the catalog. Defaults to True.
            NOTE: if an RCS value exists and is "LARGE" or "MEDIUM" for the debris object, it will be kept

        Example:

            In: TLESatelliteArray.from_file(filepath=LOCAL_SPACETRACK_CATALOG_FILE, filter_str="starlink-[1-2][4-8]99$")

            Out: ['STARLINK-2699', 'STARLINK-2499', 'STARLINK-1799', 'STARLINK-1499']

        """

        def keep_this_object_name(re_keep: str, object_name: str):
            if isinstance(re_keep, re.Pattern):
                # filter_str given, return True if object name matches otherwise return False
                return bool(re_keep.search(object_name))
            else:
                # No filter_str given, keep all objects.
                return True

        re_keep = None
        if filter_str:
            re_keep = re.compile(filter_str, re.IGNORECASE)

        file_ext = filepath.suffix
        satellites = []
        names = []
        cat_ids = []
        if file_ext == ".xml":
            parse_func = omm.parse_xml
            read_mode = "r"
        elif file_ext == ".pkl":
            parse_func = pickle.load
            read_mode = "rb"

        print(f"SatelliteArray.from_file(): Loading {filepath}")
        with open(filepath, mode=read_mode) as f:
            for fields in parse_func(f):
                # Skip small or unknown RCS debris objects
                if (
                    exclude_small_debris
                    and "OBJECT_TYPE" in fields
                    and str(fields["OBJECT_TYPE"]) == "DEBRIS"
                ):
                    if "RCS_SIZE" in fields and not (
                        str(fields["RCS_SIZE"]) == "LARGE"
                        or str(fields["RCS_SIZE"]) == "MEDIUM"
                    ):
                        continue

                # Skip filtered objects.
                if not keep_this_object_name(
                    re_keep=re_keep, object_name=fields["OBJECT_NAME"]
                ):
                    continue

                this_sat = api.Satrec()
                omm.initialize(sat=this_sat, fields=fields)
                satellites.append(this_sat)
                names.append(fields["OBJECT_NAME"])
                cat_ids.append(int(fields["NORAD_CAT_ID"]))

        if len(satellites) == 0:
            err_msg = "No satellites were found in catalog."
            if re_keep is not None:
                err_msg += f" Searched for object names containing regex 'filter_str' value of '{filter_str}'."
            raise RuntimeError(err_msg)

        return cls(satellites, names, cat_ids)

    @property
    def size(self) -> int:
        """Returns the number of satellites in the array."""
        return self._size

    @property
    def names(self):
        """Returns the names of the satellites in the array."""
        return self._names

    def name(self, i: int) -> str:
        """Returns name of satellite at index i in the TleSatelliteArray

        Args:
            i (int): Index of satellite in TleSatelliteArray

        Returns:
            str: Name of satellite
        """
        return self.names[i]

    @property
    def cat_ids(self):
        """Returns the NORAD catalog IDs of the satellites in the array."""
        return self._cat_ids

    def cat_id(self, i: int) -> int:
        """Returns NORAD catalog ID of satellite at index i in the TleSatelliteArray

        Args:
            i (int): Index of satellite in TleSatelliteArray

        Returns:
            int: NORAD catalog ID of satellite
        """
        return self.cat_ids[i]

    def propagate(
        self, epoch: Epoch, to_frame: ReferenceFrame = SpiceFrame("EARTH", "J2000")
    ) -> tuple[Position, Velocity, int]:
        """
        Computes the satellite's position and velocity in ECIJ2000 at a specified time.

        Args:
            epoch (Epoch): The target time as an Epoch object.
            to_frame (ReferenceFrame): The target reference frame for the propagated state.

        Returns:
            tuple: A tuple containing:
                - r (list): Position vector [x, y, z] in meters in the ECIJ2000 frame.
                - v (list): Velocity vector [vx, vy, vz] in meters per second in the ECIJ2000 frame.
                - e (int): Error code from the SGP4 propagation (0 indicates success).
        """
        jd1, jd2 = epoch.jd_utc_split()
        e, r, v = self._satrec_array.sgp4(jd=np.array([jd1]), fr=np.array([jd2]))
        # Assume that this is always a single time step and squeeze the arrays
        e = e.squeeze()
        r_teme = r.squeeze()
        v_teme = v.squeeze()

        # Convert from TEME to ECI
        frame_transform = self._teme_frame.get_transform_to(to_frame, epoch)
        r_eci = frame_transform.apply_to_position(r_teme)
        v_eci = frame_transform.apply_to_velocity(v_teme)

        return (
            (r_eci * 1000.0),
            (v_eci * 1000.0),
            e,
        )
