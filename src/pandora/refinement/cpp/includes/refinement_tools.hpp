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
This module contains functions used by the Refinement algorithms in cpp.
*/

#ifndef REFINEMENT_TOOLS_HPP
#define REFINEMENT_TOOLS_HPP

#include <string>
#include <tuple>

/**
 * @brief Validate costs and return early if necessary.
 *
 * @param cost_0 Cost at disp - 1.
 * @param cost_1 Cost at disp.
 * @param cost_2 Cost at disp + 1.
 * @param measure The measure used to create the cost volume ("min" or "max").
 * @return A tuple containing: whether the costs are valid, cost_0, cost_1, cost_2,
 * inverse_cost_0, inverse_cost_1, inverse_cost_2.
 */
std::tuple<bool, float, float, float, float, float, float> validate_costs_and_get_variables(
    float cost_0,
    float cost_1,
    float cost_2,
    const std::string& measure
);


#endif // REFINEMENT_TOOLS_HPP