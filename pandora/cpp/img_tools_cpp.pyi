# pylint: skip-file
import numpy as np
from typing import Tuple

def find_valid_neighbors(dirs, disp, valid, row, col, msk_pixel_invalid): ...
def interpolate_nodata_sgm(
    img: np.ndarray, valid: np.ndarray, msk_pixel_invalid: int, msk_pixel_filled_nodata: int
) -> Tuple[np.ndarray, np.ndarray]: ...
