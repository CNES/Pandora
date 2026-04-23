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
def compute_matching_costs(img_left, imgs_right, cv, disps, census_width, census_height):
    """
    Given a left image and right images (multiple when doing subpixellic), compute the Census matching costs for all disparities, with the given window.

    :param img_left: the left image
    :type img_left: 2D np.array (row, col) dtype = np.float32
    :param imgs_right: the right images
    :type imgs_right: List of 2D np.array (row, col) dtype = np.float32
    :param cv: cost volume to fill
    :type cv: 3D np.array (row, col, disps) dtype = np.float32
    :param disps: the disparities to sample, sorted
    :type disps: np.array (disps) dtype = np.float32
    :param census_width: the width of the census window
    :type census_width: int
    :param census_height: the height of the census window
    :type census_height: int
    :return: the filled cost volume
    :rtype: 3D np.array (row, col, disps) dtype = np.float32
    """
    ...

def reverse_cost_volume(left_cv, disp_min):
    """
    Create the right_cv from the left_one by reindexing (i,j,d) -> (i, j + d, -d)
    :param left_cv: the 3D cost_colume data array, with dimensions row, col, disp
    :type left_cv: np.ndarray(dtype=float32)
    :param disp_min: the minimum of the right disparities
    :type min_disp: int64
    :return: The right cost volume data
    :rtype: 3D np.ndarray of type float32
    """
    ...

def reverse_disp_range(left_min, left_max):
    """
    Create the right disp ranges from the left disp ranges
    :param left_min: the 2D left disp min array, with dimensions row, col
    :type left_min: np.ndarray(dtype=float32)
    :param left_max: the 2D left disp max array, with dimensions row, col
    :type left_max: np.ndarray(dtype=float32)
    :return: The min and max disp ranges for the right image
    :rtype: Tuple[np.ndarray(dtype=float32), np.ndarray(dtype=float32)]
    """
    return None, None

def cv_masked(
    cost_volume, mask_left, mask_right_shift, mask_right_native, disp_min, disp_max, disp_range, global_disp_min, subpix
):
    """
    Apply masking to cost volume based on valid pixels and local disparity ranges

    Masks the cost volume by:
    - Setting entire disparity range to NaN for pixels masked in left image
    - Setting disparities outside local [disp_min, disp_max] range to NaN
    - Setting disparities to NaN if corresponding right pixel is masked

    :param cost_volume: the 3D cost volume data array (row, col, disp)
    :type cost_volume: np.ndarray(dtype=float32)
    :param mask_left: the 2D left mask array (row, col)
    :type mask_left: np.ndarray(dtype=float32)
    :param mask_right_native: the 2D right mask for whole pixel disparities (row, col)
    :type mask_right_native: np.ndarray(dtype=float32)
    :param mask_right_shift: the 2D right mask for subpix disparities (row, col)
    :type mask_right_shift: np.ndarray(dtype=float32)
    :param disp_min: the 2D local minimum disparities (row, col)
    :type disp_min: np.ndarray(dtype=float32)
    :param disp_max: the 2D local maximum disparities (row, col)
    :type disp_max: np.ndarray(dtype=float32)
    :param disp_range: the 1D disparity range values (disp)
    :type disp_range: np.ndarray(dtype=float32)
    :param global_disp_min: global cost volume minimum disparity
    :type global_disp_min: int
    :param subpix: subpixel precision
    :type subpix: int

    :return: None
    :rtype: None
    """
    ...
