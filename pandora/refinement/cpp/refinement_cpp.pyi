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
from typing import Tuple

def quadratic_refinement_method(cost, disp, measure, cst_pandora_msk_pixel_stopped_interpolation):
    """
    Return the subpixel disparity and cost, by fitting a quadratic curve

    :param cost: cost of the values disp - 1, disp, disp + 1
    :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
    :param disp: the disparity
    :type disp: float
    :param measure: the type of measure used to create the cost volume
    :param measure: string = min | max
    :param cst_pandora_msk_pixel_stopped_interpolation: value for the PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION \
    constant in pandora.constants
    :param cst_pandora_msk_pixel_stopped_interpolation: int
    :return: the disparity shift, the refined cost and the state of the pixel ( Information: \
    calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
    :rtype: float, float, int
    """
    ...

def loop_refinement(
    cv,
    disp,
    mask,
    d_min,
    d_max,
    subpixel,
    measure,
    method,
    cst_pandora_msk_pixel_invalid,
    cst_pandora_msk_pixel_stopped_interpolation,
):
    """
    Apply for each pixels the refinement method

    :param cv: cost volume to refine
    :type cv: 3D numpy array (row, col, disp)
    :param disp: disparity map
    :type disp: 2D numpy array (row, col)
    :param mask: validity mask
    :type mask: 2D numpy array (row, col)
    :param d_min: minimal disparity
    :type d_min: int
    :param d_max: maximal disparity
    :type d_max: int
    :param subpixel: subpixel precision used to create the cost volume
    :type subpixel: int ( 1 | 2 | 4 )
    :param measure: the measure used to create the cot volume
    :param measure: string
    :param method: the refinement method
    :param method: function
    :param cst_pandora_msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param cst_pandora_msk_pixel_invalid: int
    :param cst_pandora_msk_pixel_stopped_interpolation: value for the PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION \
    constant in pandora.constants
    :param cst_pandora_msk_pixel_stopped_interpolation: int
    :return: the refine coefficient, the refine disparity map, and the validity mask
    :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
    """
    ...

def loop_approximate_refinement(
    cv,
    disp,
    mask,
    d_min,
    d_max,
    subpixel,
    measure,
    method,
    cst_pandora_msk_pixel_invalid,
    cst_pandora_msk_pixel_stopped_interpolation,
):
    """
    Apply for each pixels the refinement method on the right disparity map which was created with the \
    approximate method : a diagonal search for the minimum on the left cost volume

    :param cv: the left cost volume
    :type cv: 3D numpy array (row, col, disp)
    :param disp: right disparity map
    :type disp: 2D numpy array (row, col)
    :param mask: right validity mask
    :type mask: 2D numpy array (row, col)
    :param d_min: minimal disparity
    :type d_min: int
    :param d_max: maximal disparity
    :type d_max: int
    :param subpixel: subpixel precision used to create the cost volume
    :type subpixel: int ( 1 | 2 | 4 )
    :param measure: the type of measure used to create the cost volume
    :type measure: string = min | max
    :param method: the refinement method
    :type method: function
    :param cst_pandora_msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param cst_pandora_msk_pixel_invalid: int
    :param cst_pandora_msk_pixel_stopped_interpolation: value for the PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION \
    constant in pandora.constants
    :param cst_pandora_msk_pixel_stopped_interpolation: int
    :return: the refine coefficient, the refine disparity map, and the validity mask
    :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
    """
    ...

def vfit_refinement_method(
    cost, disp, measure, cst_pandora_msk_pixel_stopped_interpolation
) -> Tuple[float, float, int]:
    """
    Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)

    :param cost: cost of the values disp - 1, disp, disp + 1
    :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
    :param disp: the disparity
    :type disp: float
    :param measure: the type of measure used to create the cost volume
    :param measure: string = min | max
    :param cst_pandora_msk_pixel_stopped_interpolation: value for the PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION \
    constant in pandora.constants
    :param cst_pandora_msk_pixel_stopped_interpolation: int
    :return: the disparity shift, the refined cost and the state of the pixel( Information: calculations \
    stopped at the pixel step, sub-pixel interpolation did not succeed )
    :rtype: float, float, int
    """
    ...
