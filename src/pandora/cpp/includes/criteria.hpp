/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA
 *
 *     https://github.com/CNES/Pandora
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This module contains functions associated to the Criteria in cpp.
*/

#ifndef CRITERIA_HPP
#define CRITERIA_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Find pixels whose disparity range contains one or more invalid pixels
 *
 * @param disps disparity bounds for each pixel
 * @param img_mask mask of pixels considered as nan / invalid
 * @return a mask with affected pixels as true, and unaffected pixels as false
 */
py::array_t<bool> partially_missing_variable_ranges(
    py::array_t<float> disps,
    py::array_t<bool> img_mask
);

/**
 * @brief Allocate right image mask flags on the validity mask
 *
 * For each left pixel, checks whether the full disparity interval in the right
 * image is invalidated by the right validity mask or by nodata (dilated).
 *
 * @param validity_mask validity mask to update in-place
 * @param bit_1_skip columns to skip (True = already flagged outside image)
 * @param nodata_dil_mask dilated nodata mask of the right image
 * @param right_mask right validity mask (0 = valid, 1 = invalid)
 * @param disp_min minimum disparity
 * @param disp_max maximum disparity
 * @param offset row/col offset of the cost volume
 * @param flag_validity_right flag for PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
 * @param flag_nodata_right flag for PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
 */
void allocate_right_mask_cpp(
    py::array_t<uint16_t> validity_mask,
    py::array_t<bool> bit_1_skip,
    py::array_t<bool> nodata_dil_mask,
    py::array_t<uint8_t> right_mask,
    int disp_min,
    int disp_max,
    int offset,
    uint16_t flag_validity_right,
    uint16_t flag_nodata_right
);

#endif  // CRITERIA_HPP