from typing import Tuple

import numpy as np

from rsalib import Epoch
from rsalib.frames import CustomFrame, FrameTransform, SpiceFrame
from rsalib.utils.vallado_utils import precess, truemean


class TEME(CustomFrame):
    def __repr__(self) -> str:
        return f"TEME(DATUM={self.get_datum()})"

    def __init__(self):
        self._datum = SpiceFrame("EARTH", "J2000")  # Defined relative to ECIJ2000

    def get_datum(self) -> SpiceFrame:
        return self._datum

    def get_transform_to_datum(self, time: Epoch) -> Tuple[FrameTransform, SpiceFrame]:
        jd1, jd2 = time.jd_utc_split()
        ttt = (jd1 + jd2 - 2451545.0) / 36525.0  # Julian centuries since J2000.0

        order = 10
        eqeterms = 2
        opt = "a"

        prec, _, _, _, _ = precess(ttt, "80")
        _, _, _, _, _, nutteme = truemean(ttt, order, eqeterms, opt)

        teme2eci_matrix = prec @ nutteme

        transform = FrameTransform(
            self,
            self._datum,
            translation=np.zeros(3),
            rotation=teme2eci_matrix,
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        return transform, self._datum
