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
This module contains functions associated to the Interval Bounds algorithms in cpp.
*/

#ifndef INTERVAL_BOUNDS_HPP
#define INTERVAL_BOUNDS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Computes interval bounds on the disparity.
 *
 * @param cv cost volume
 * @param disp_interval disparity data
 * @param possibility_threshold possibility threshold used for interval computation
 * @param type_factor either 1 or -1; used to adapt the possibility computation to max or
 * min measures
 * @param grids array containing min and max disparity grids
 * @param disparity_range array containing disparity range
 * @return the infimum and supremum (not regularized) of the set containing the true disparity
 */
std::tuple<py::array_t<float>, py::array_t<float>> compute_interval_bounds(
    py::array_t<float> cv,
    py::array_t<float> disp_interval,
    float possibility_threshold,
    float type_factor,
    py::array_t<int> grids,
    py::array_t<float> disparity_range

);

#endif  // INTERVAL_BOUNDS_HPP