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
def interpolate_occlusion_sgm(disp, valid, msk_pixel_occlusion, msk_pixel_filled_occlusion, msk_pixel_invalid):
    """
    Interpolation of the left disparity map to resolve occlusion conflicts.
    Interpolate occlusion by moving by selecting
    the right lowest value along paths from 8 directions.

    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param msk_pixel_occlusion: value for the PANDORA_MSK_PIXEL_OCCLUSION constant in pandora.constants
    :param msk_pixel_occlusion: int
    :param msk_pixel_filled_occlusion: value for the PANDORA_MSK_PIXEL_FILLED_OCCLUSION constant in pandora.constants
    :param msk_pixel_filled_occlusion: int
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param msk_pixel_invalid: int
    :return: the interpolate left disparity map, with the validity mask update :

        - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
    :rtype: : tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None

def interpolate_mismatch_sgm(
    disp, valid, msk_pixel_mismatch, msk_pixel_filled_mismatch, msk_pixel_occlusion, msk_pixel_invalid
):
    """
    Interpolation of the left disparity map to resolve mismatch conflicts. Interpolate mismatch by finding the
    nearest correct pixels in 8 different directions and use the median of their disparities.
    Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions.

    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param msk_pixel_mismatch: value for the PANDORA_MSK_PIXEL_MISMATCH constant in pandora.constants
    :param msk_pixel_mismatch: int
    :param msk_pixel_filled_mismatch: value for the PANDORA_MSK_PIXEL_FILLED_MISMATCH constant in pandora.constants
    :param msk_pixel_filled_mismatch: int
    :param msk_pixel_occlusion: value for the PANDORA_MSK_PIXEL_OCCLUSION constant in pandora.constants
    :param msk_pixel_occlusion: int
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param msk_pixel_invalid: int
    :return: the interpolate left disparity map, with the validity mask update :

        - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None

def interpolate_occlusion_mc_cnn(disp, valid, msk_pixel_occlusion, msk_pixel_filled_occlusion, msk_pixel_invalid):
    """
    Interpolation of the left disparity map to resolve occlusion conflicts.
    Interpolate occlusion by moving left until
    we find a position labeled correct.

    Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
    patches. The journal of machine learning research, 17(1), 2287-2318.

    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param msk_pixel_occlusion: value for the PANDORA_MSK_PIXEL_OCCLUSION constant in pandora.constants
    :param msk_pixel_occlusion: int
    :param msk_pixel_filled_occlusion: value for the PANDORA_MSK_PIXEL_FILLED_OCCLUSION constant in pandora.constants
    :param msk_pixel_filled_occlusion: int
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param msk_pixel_invalid: int
    :return: the interpolate left disparity map, with the validity mask update :

        - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None

def interpolate_mismatch_mc_cnn(disp, valid, msk_pixel_mismatch, msk_pixel_filled_mismatch, msk_pixel_invalid):
    """
    Interpolation of the left disparity map to resolve mismatch conflicts.
    Interpolate mismatch by finding the nearest
    correct pixels in 16 different directions and use the median of their disparities.

    Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
    patches. The journal of machine learning research, 17(1), 2287-2318.

    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param msk_pixel_mismatch: value for the PANDORA_MSK_PIXEL_MISMATCH constant in pandora.constants
    :param msk_pixel_mismatch: int
    :param msk_pixel_filled_mismatch: value for the PANDORA_MSK_PIXEL_FILLED_MISMATCH constant in pandora.constants
    :param msk_pixel_filled_mismatch: int
    :param msk_pixel_invalid: value for the PANDORA_MSK_PIXEL_INVALID constant in pandora.constants
    :param msk_pixel_invalid: int
    :return: the interpolate left disparity map, with the validity mask update :

        - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None
