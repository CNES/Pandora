# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

def partially_missing_variable_ranges(disps: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Returns a mask of the pixels with a partially missing variable range in the right image.

    :param disps: Disparity range, np.ndarray of shape (2, width, height)
    :type disps: np.ndarray(float)
    :param img_mask: Mask of valid pixels in the right image, np.ndarray of shape (width, height)
    :type img_mask: np.ndarray(bool)
    """
    ...

def allocate_right_mask_cpp(
    validity_mask: np.ndarray,
    bit_1_skip: np.ndarray,
    nodata_dil_mask: np.ndarray,
    right_mask: np.ndarray,
    disp_min: int,
    disp_max: int,
    offset: int,
    flag_validity_right: int,
    flag_nodata_right: int,
) -> None:
    """
    Allocate right image mask flags on the validity mask (in-place).

    :param validity_mask: Validity mask, np.ndarray of shape (row, col)
    :type validity_mask: np.ndarray(uint16)
    :param bit_1_skip: Columns to skip, np.ndarray of shape (col,)
    :type bit_1_skip: np.ndarray(bool)
    :param nodata_dil_mask: Dilated nodata mask, np.ndarray of shape (row, col)
    :type nodata_dil_mask: np.ndarray(bool)
    :param right_mask: Right validity mask (0=valid, 1=invalid), np.ndarray of shape (row, col)
    :type right_mask: np.ndarray(uint8)
    :param disp_min: Minimum disparity
    :type disp_min: int
    :param disp_max: Maximum disparity
    :type disp_max: int
    :param offset: Row/col offset of the cost volume
    :type offset: int
    :param flag_validity_right: PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT flag value
    :type flag_validity_right: int
    :param flag_nodata_right: PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING flag value
    :type flag_nodata_right: int
    """
    ...
