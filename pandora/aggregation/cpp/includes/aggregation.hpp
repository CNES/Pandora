/* Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the Aggregation algorithms in cpp.
*/

#ifndef CBCA_HPP
#define CBCA_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Build the fully aggregated matching cost for one
 * disparity, E = S_v(row, col + bottom_arm_length) - S_v(row, col - top_arm_length - 1)
 *
 * @param input cost volume for the current disparity
 * @param step1 horizontal integral image from the cbca_step1, with an extra column that contains 0
 * @param cross_left cross support of the left image
 * @param cross_right cross support of the right image
 * @param range_row left column for the current disparity (i.e : np.arrange(nb columns), where the
 * correspondent in the right image is reachable)
 * @param range_row_right: right column for the current disparity 
 * (i.e : np.arrange(nb columns) - disparity, where column - disparity >= 0 and <= nb columns)
 * @return the fully aggregated matching cost, and the total number of support pixels used for the
 * aggregation
 */
std::tuple<py::array_t<float>, py::array_t<float>> cbca(
    py::array_t<float> input,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_row,
    py::array_t<int64_t> range_row_right
);

/**
 * @brief Compute the cross support for an image: find the 4 arms.
 * Enforces a minimum support region of 3×3 if pixels are valid.
 * The cross support of invalid pixels (pixels that are np.inf) is 0 for the 4 arms.
 *
 * @param image input image
 * @param len_arms maximal length of arms
 * @param intensity maximal intensity difference allowed for neighboring pixels
 * @return a 3D array with the four arm lengths computed for each pixel
 */
py::array_t<int16_t> cross_support(py::array_t<float> image, int16_t len_arms, float intensity);

#endif  // CBCA_HPP