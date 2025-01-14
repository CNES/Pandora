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
This module contains functions associated to the Interval Tools in cpp.
*/

#ifndef INTERVAL_TOOLS_HPP
#define INTERVAL_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Create a boolean connection matrix from segment coordinates.
 *
 * @param border_left array containing the coordinates of the left border of segments
 * @param border_right array containing the coordinates of the right border of segments
 * @param depth the depth for regularization; corresponds to the number of rows to explore 
 * above and below
 * @return a symmetric boolean matrix indicating whether the segments are connected
 */
py::array_t<bool> create_connected_graph(
    py::array_t<int> border_left, py::array_t<int> border_right, int depth
);

/**
 * @brief Regularize intervals based on quantiles and a connection graph.
 *
 * @param interval_inf lower bound of the disparity to regularize
 * @param interval_sup upper bound of the disparity to regularize
 * @param border_left array containing the coordinates of the left border of segments
 * @param border_right array containing the coordinates of the right border of segments
 * @param connection_graph matrix indicating whether the segments are connected
 * @param quantile quantile value for the regularized output (0 <= quantile <= 1)
 * @return tuple of regularized lower and upper bounds of the disparity, and a boolean mask 
 * indicating regularization
 */
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<bool>> graph_regularization(
    py::array_t<float> interval_inf,
    py::array_t<float> interval_sup,
    py::array_t<int> border_left,
    py::array_t<int> border_right,
    py::array_t<bool> connection_graph,
    float quantile
);

#endif  // INTERVAL_TOOLS_HPP