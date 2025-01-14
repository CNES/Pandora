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
 * @brief Given the matching cost for one disparity, build a horizontal integral image storing
 * the cumulative row sum, S_h(row, col) = S_h(row-1, col) + cv(row, col)
 *
 * @param input cost volume for the current disparity
 * @return the horizontal integral image, step 1
 */
py::array_t<float> cbca_step_1(py::array_t<float> input);

/**
 * @brief Given the horizontal integral image, computed the horizontal matching cost for one 
 * disparity, E_h(row, col) = S_h(row + right_arm_length, col) - S_h(row - left_arm_length -1, col)
 *
 * @param step1 horizontal integral image from the cbca_step1, with an extra column that contains 0
 * @param cross_left cross support of the left image
 * @param cross_right cross support of the right image
 * @param range_col left column for the current disparity (i.e : np.arrange(nb columns), where the
 * correspondent in the right image is reachable)
 * @param range_col_right: right column for the current disparity 
 * (i.e : np.arrange(nb columns) - disparity, where column - disparity >= 0 and <= nb columns)
 * @return the horizontal matching cost for the current disparity, and the number of support pixels 
 * used for the step 2
 */
std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_2(
    py::array_t<float> step1,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
);

/**
 * @brief Given the horizontal matching cost, build a vertical integral image for one disparity,
 * S_v = S_v(row, col - 1) + E_h(row, col)
 *
 * @param step2 horizontal matching cost, from the cbca_step2
 * @return the vertical integral image for the current disparity
 */
py::array_t<float> cbca_step_3(py::array_t<float> step2);

/**
 * @brief Given the vertical integral image, build the fully aggregated matching cost for one
 * disparity, E = S_v(row, col + bottom_arm_length) - S_v(row, col - top_arm_length - 1)
 *
 * @param step3 vertical integral image, from the cbca_step3, with an extra row that contains 0
 * @param sum2 the number of support pixels used for the step 2
 * @param cross_left cross support of the left image
 * @param cross_right cross support of the right image
 * @param range_col left column for the current disparity (i.e : np.arrange(nb columns), where the
 * correspondent in the right image is reachable)
 * @param range_col_right right column for the current disparity 
 * (i.e : np.arrange(nb columns) - disparity, where column - disparity >= 0 and <= nb columns)
 * @return the fully aggregated matching cost, and the total number of support pixels used for the
 * aggregation
 */
std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_4(
    py::array_t<float> step3,
    py::array_t<float> sum2,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
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