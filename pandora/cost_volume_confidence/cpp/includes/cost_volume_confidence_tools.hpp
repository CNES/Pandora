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
This module contains functions associated to the general Cost Volume Confidence algorithms in cpp.
*/

#ifndef COST_VOLUME_CONFIDENCE_TOOLS_HPP
#define COST_VOLUME_CONFIDENCE_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Find the index where an element should be inserted to maintain order.
 * The returned index i satisfies array[i-1] < value <= array[i]
 *
 * @param array the 1D array where value would be inserted
 * @param value the value to insert in array
 * @return the index where value should be inserted
 */
size_t searchsorted(const py::array_t<float>& array, float value);
std::tuple<float, float,
           pybind11::detail::unchecked_mutable_reference<float, 2>,
           pybind11::detail::unchecked_mutable_reference<float, 2>>
min_max_cost(
    py::detail::unchecked_reference<float, 3> r_cv,
    size_t n_row,
    size_t n_col,
    size_t n_disp
);
#endif  // COST_VOLUME_CONFIDENCE_TOOLS_HPP