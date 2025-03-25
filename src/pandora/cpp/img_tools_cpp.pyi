# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
import numpy as np
from typing import Tuple

def find_valid_neighbors(dirs, disp, valid, row, col, msk_pixel_invalid):
    """
    Find valid neighbors along directions

    :param dirs: directions in which the valid neighbors will be searched, around the (row, col) pixel. \
    Ex: [[0,1],[1,0],[0,-1],[-1,0]] to search in the axis-aligned cross centered on the pixel.
    :type dirs: 2D np.ndarray (row, col)
    :param disp: disparity map
    :type disp: 2D np.ndarray (row, col)
    :param valid: validity mask
    :type valid: 2D np.ndarray (row, col)
    :param row: row current value
    :type row: int
    :param col: col current value
    :type col: int
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :type msk_pixel_invalid: int
    :return: valid neighbors
    :rtype: 2D np.ndarray
    """
    ...

def interpolate_nodata_sgm(
    img: np.ndarray, valid: np.ndarray, msk_pixel_invalid: int, msk_pixel_filled_nodata: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolation of the input image to resolve invalid (nodata) pixels.
    Interpolate invalid pixels by finding the nearest correct pixels in 8 different directions
    and use the median of their disparities.

    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

    :param img: input image
    :type img: 2D np.ndarray (row, col)
    :param valid: validity mask
    :type valid: 2D np.ndarray (row, col)
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :type msk_pixel_invalid: int
    :param msk_pixel_filled_nodata: value for the PANDORA_MSK_PIXEL_FILLED_NODATA constant in pandora.constants
    :type msk_pixel_filled_nodata: int
    :return: the interpolate input image, with the validity mask update :

        - If out & PANDORA_MSK_PIXEL_FILLED_NODATA != 0 : Invalid pixel : filled nodata pixel
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None
