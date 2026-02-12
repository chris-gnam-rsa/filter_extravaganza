import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import spiceypy as spice
from platformdirs import user_cache_dir

from rsalib import KERNELS_PATH

#######################
### Remote Kernels  ###
#######################
CACHE_DIR = Path(user_cache_dir("rsalib")) / "data/kernels"

# Large kernels that should be downloaded rather than bundled
REMOTE_KERNELS = {
    "de440s.bsp": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        "sha256": None,
    },
    "earth_200101_990827_predict.bpc": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/earth_200101_990827_predict.bpc",
        "sha256": None,
    },
    "earth_620120_240827.bpc": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/earth_620120_240827.bpc",
        "sha256": None,
    },
}


def _get_kernel_path(relative_path: Path) -> Path:
    """
    Resolve a kernel path, downloading if necessary.

    For bundled kernels, returns KERNELS_PATH / relative_path.
    For remote kernels, downloads to cache and returns cached path.
    """
    filename = relative_path.name

    # Check if it's a remote kernel
    if filename in REMOTE_KERNELS:
        return _ensure_remote_kernel(filename)

    # Otherwise it's bundled
    return KERNELS_PATH / relative_path


def _ensure_remote_kernel(name: str) -> Path:
    """Download a remote kernel if needed and return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = CACHE_DIR / name
    info = REMOTE_KERNELS[name]

    if not path.exists():
        print(f"Downloading {name}...")

        # Create SSL context with certifi's certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Download with proper SSL
        request = urllib.request.Request(info["url"])
        with urllib.request.urlopen(request, context=ssl_context) as response:
            with open(path, "wb") as f:
                f.write(response.read())

        if info.get("sha256"):
            _verify_hash(path, info["sha256"])

        print(f"Cached at {path}")

    return path


def _verify_hash(path: Path, expected: str) -> None:
    """Verify file integrity via SHA256 hash."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    if sha256.hexdigest() != expected:
        path.unlink()
        raise ValueError(f"Hash mismatch for {path.name}. File deleted.")


#######################
### Kernel Mappings ###
#######################
lsk_kernel = KERNELS_PATH / "lsk/naif0012.tls"

_BUILTIN_ORIGINS = ["SSB"]  # Solar System Barycenter
_INCLUDED_ORIGINS = {
    Path("spk/de440s.bsp"): [  # Now using relative paths
        "MERCURY BARYCENTER",
        "VENUS BARYCENTER",
        "EARTH MOON BARYCENTER",
        "MARS BARYCENTER",
        "JUPITER BARYCENTER",
        "SATURN BARYCENTER",
        "URANUS BARYCENTER",
        "NEPTUNE BARYCENTER",
        "PLUTO BARYCENTER",
        "SUN",
        "MERCURY",
        "VENUS",
        "MOON",
        "EARTH",
    ]
}

_BUILTIN_ORIENTATIONS = ["J2000", "ICRF", "ECLIPJ2000", "GALACTIC"]
_INCLUDED_ORIENTATIONS = {
    Path("pck/pck00011.tpc"): [  # Bundled
        "IAU_MERCURY",
        "IAU_VENUS",
        "IAU_EARTH",
        "IAU_MARS",
        "IAU_JUPITER",
        "IAU_SATURN",
        "IAU_URANUS",
        "IAU_NEPTUNE",
        "IAU_MOON",
    ],
    Path("pck/earth_200101_990827_predict.bpc"): ["ITRF93"],  # Remote
    Path("pck/earth_620120_240827.bpc"): ["ITRF93"],  # Remote
}


def _build_origin_lookup():
    lookup = {}

    for body_name in _BUILTIN_ORIGINS:
        body_id = spice.bodn2c(body_name)
        lookup[body_id] = []

    for kernel_rel_path, names in _INCLUDED_ORIGINS.items():
        for body_name in names:
            body_id = spice.bodn2c(body_name)
            if body_id not in lookup:
                lookup[body_id] = []
            lookup[body_id].append(kernel_rel_path)

    return lookup


_ORIGIN_KERNELS = _build_origin_lookup()


def _build_orientation_lookup():
    lookup = {}

    for frame in _BUILTIN_ORIENTATIONS:
        lookup[frame] = []

    for kernel_rel_path, frames in _INCLUDED_ORIENTATIONS.items():
        for frame in frames:
            if frame not in lookup:
                lookup[frame] = []
            lookup[frame].append(kernel_rel_path)

    return lookup


_ORIENTATION_KERNELS = _build_orientation_lookup()


##############################
### Kernel Loading Support ###
##############################


def _is_kernel_loaded(kernel_path):
    """Check if a specific kernel file is already loaded."""
    normalized = os.path.normpath(os.path.abspath(kernel_path))
    count = spice.ktotal("ALL")
    for i in range(count):
        file, ftype, source, handle = spice.kdata(i, "ALL")
        if os.path.normpath(os.path.abspath(file)) == normalized:
            return True
    return False


def _safe_load_kernel(kernel_rel_path):
    """Load a kernel by relative path, downloading if necessary."""
    if isinstance(kernel_rel_path, str):
        kernel_rel_path = Path(kernel_rel_path)

    path = _get_kernel_path(kernel_rel_path)

    if not path.exists():
        raise FileNotFoundError(f"Kernel not found: {path}")
    if not _is_kernel_loaded(path):
        spice.furnsh(str(path))


def load_kernels(kernel_list):
    if kernel_list is None:
        return

    if isinstance(kernel_list, (str, Path)):
        _safe_load_kernel(kernel_list)
    elif isinstance(kernel_list, list):
        for kernel in kernel_list:
            _safe_load_kernel(kernel)
    else:
        raise TypeError("kernel_list must be a string, Path, or list.")


def unload_all_kernels():
    spice.kclear()


def clear_kernel_cache():
    """Remove all cached (downloaded) kernel files."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            f.unlink()
        print(f"Cleared kernel cache at {CACHE_DIR}")


#################################
### Leap Second Kernel Loader ###
#################################


def _load_lsk():
    if not spice.expool("DELTET/DELTA_AT"):
        print(
            "\nNo LSK kernel was provided by the user:\n",
            "   - A default kernel was found\n",
            "   - Loading default kernels: ",
            lsk_kernel,
        )
        _safe_load_kernel(Path("lsk/naif0012.tls"))


########################
### Helper Functions ###
########################
def _normalize_body(body):
    """Convert any body identifier to (naif_id, canonical_name)."""
    if isinstance(body, int):
        code = body
    else:
        code = spice.bodn2c(body.upper().strip())
    name = spice.bodc2n(code)
    return code, name


def _normalize_frame(frame):
    """Normalize frame name."""
    return frame.upper().strip()


def _has_origin_data(origin_id):
    """Check if SPK data is loaded for this body."""
    if origin_id == 0:  # SSB is always available
        return True
    try:
        et = spice.str2et("2000-01-01T12:00:00")
        spice.spkpos(spice.bodc2n(origin_id), et, "J2000", "NONE", "SSB")
        return True
    except Exception:
        return False


def _has_orientation_data(frame_name):
    """Check if frame orientation data is available."""
    if frame_name.upper() in ("J2000", "ICRF", "ECLIPJ2000", "GALACTIC"):
        return True
    try:
        et = spice.str2et("2000-01-01T12:00:00")
        spice.pxform("J2000", frame_name, et)
        return True
    except Exception:
        return False


def _load_required_kernels(origin_id, origin_name, orientation):
    """Load kernels required for this frame, only if not already available."""

    _load_lsk()

    if origin_id != 0 and not _has_origin_data(origin_id):
        origin_kernels = _ORIGIN_KERNELS.get(origin_id)
        if origin_kernels is None:
            raise ValueError(
                f"No SPK data loaded for '{origin_name}' (ID {origin_id}) "
                f"and no default kernels available."
            )
        if origin_kernels:
            print(
                f"\nFor origin '{origin_name}' (ID {origin_id}):\n",
                "   - No SPK (.bsp) was provided by the user\n",
                "   - Loading default kernels:",
                [str(k) for k in origin_kernels],
            )
            load_kernels(origin_kernels)

    if not _has_orientation_data(orientation):
        orientation_kernels = _ORIENTATION_KERNELS.get(orientation)
        if orientation_kernels is None:
            raise ValueError(
                f"No data loaded for frame '{orientation}' "
                f"and no default kernels available."
            )
        if orientation_kernels:
            print(
                f"\nFor orientation frame '{orientation}':\n",
                "   - No frame data was loaded by the user\n",
                "   - Loading default kernels:",
                [str(k) for k in orientation_kernels],
            )
            load_kernels(orientation_kernels)
