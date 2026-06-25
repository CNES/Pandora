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

#include "refinement_tools.hpp"
#include <cmath>

std::tuple<bool, float, float, float, float, float, float> validate_costs_and_get_variables(
    float cost_0,
    float cost_1,
    float cost_2,
    const std::string& measure
) {
    if (std::isnan(cost_0) || std::isnan(cost_2)) {
        // Bit 3 = 1: Information: calculations stopped at the pixel step,
        // sub-pixel interpolation did not succeed
        return {false, cost_0, cost_1, cost_2, 0.f, 0.f, 0.f};
    }

    float inverse = 1.f;
    if (measure.compare("max") == 0) { // a.compare(b) = 0 -> a = b
        // Additive inverse : if a < b then -a > -b
        inverse = -1.f;
    }

    float inverse_cost_0 = inverse * cost_0;
    float inverse_cost_1 = inverse * cost_1;
    float inverse_cost_2 = inverse * cost_2;
    // Check if cost[disp] is the minimum cost (or maximum using similarity measure) before fitting
    // If not, interpolation is not applied
    if ( inverse_cost_1 > inverse_cost_0 || inverse_cost_1 > inverse_cost_2 ) {
        return {false, cost_0, cost_1, cost_2, 0.f, 0.f, 0.f};
    }

    return {true, cost_0, cost_1, cost_2, inverse_cost_0, inverse_cost_1, inverse_cost_2};
}

