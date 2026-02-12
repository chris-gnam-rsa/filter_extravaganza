# ruff: noqa: I001, E402
# Disable ruff unsorted-imports and module top of file errors.
# Order is important for this module.

from pathlib import Path

PACKAGE_PATH = Path(__file__).parent
DATA_PATH = PACKAGE_PATH / "data"
KERNELS_PATH = DATA_PATH / "kernels"


from .epoch import Epoch

from .rotation import Rotation

from . import frames

from . import units as Units

from . import satellites

__all__ = [
    "frames",
    "satellites",
    "Epoch",
    "Rotation",
    "Units",
]
