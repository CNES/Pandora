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
This module contains functions associated with matching cost general algorithms in cpp.
*/

#ifndef MATCHING_COST_HPP
#define MATCHING_COST_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Create the right_cv from the left_one by reindexing (i,j,d) -> (i, j + d, -d)
 *
 * @param left_cv: the 3D cost_colume data array, with dimensions row, col, disp
 * @param disp_min: the minimum of the right disparities
 * @return: The right cost volume data
 */
py::array_t<float> reverse_cost_volume(
    py::array_t<float> left_cv,
    int min_disp
);

/**
 * @brief Create the right disp ranges from the left disp ranges
 *
 * @param left_min: the 2D left disp min array, with dimensions row, col
 * @param left_max: the 2D left disp min array, with dimensions row, col
 * @return: The min and max disp ranges for the right image
 */
std::tuple<py::array_t<float>, py::array_t<float>> reverse_disp_range(
    py::array_t<float> left_min,
    py::array_t<float> left_max
);

/**
 * @brief Apply masking to cost volume based on valid pixels and local disparity ranges
 *
 * Masks the cost volume by:
 * - Setting entire disparity range to NaN for pixels masked in left image
 * - Setting disparities outside local [disp_min, disp_max] range to NaN
 * - Setting disparities to NaN if corresponding right pixel is masked
 *
 * @param cost_volume: the 3D cost volume data array (row, col, disp)
 * @param mask_left: the 2D left mask array (row, col)
 * @param mask_right_native: the 2D right mask for whole pixel disparities (row, col)
 * @param mask_right_shift: the 2D right mask for subpix disparities (row, col)
 * @param disp_min: the 2D local minimum disparities (row, col)
 * @param disp_max: the 2D local maximum disparities (row, col)
 * @param disp_range: the 1D disparity range values (disp)
 * @param global_disp_min: global cost volume minimum disparity
 * @param subpix: subpixel precision
 * @return: None
 */
void cv_masked(
    py::array_t<float> cost_volume,
    py::array_t<float> mask_left,
    py::array_t<float> mask_right_native,
    py::array_t<float> mask_right_shift,
    py::array_t<float> disp_min,
    py::array_t<float> disp_max,
    py::array_t<float> disp_range,
    int global_disp_min,
    int subpix
);

#endif  // MATCHING_COST_HPP