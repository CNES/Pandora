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
This module contains functions used by the Refinement algorithms in cpp.
*/

#ifndef REFINEMENT_TOOLS_HPP
#define REFINEMENT_TOOLS_HPP

#include <pybind11/numpy.h>
#include <tuple>
#include <string>

namespace py = pybind11;

/**
 * @brief Validate costs and return early if necessary
 *
 * @param cost Array of costs for disp - 1, disp, disp + 1
 * @param measure The measure used to create the cost volume.
 * @return A tuple containing: whether the costs are valid, c0, c1, c2, ic0, ic1, ic2
 */
std::tuple<bool, float, float, float, float, float, float> validate_costs_and_get_variables(
    pybind11::array_t<float>& cost,
    const std::string& measure
);

#endif // REFINEMENT_TOOLS_HPP