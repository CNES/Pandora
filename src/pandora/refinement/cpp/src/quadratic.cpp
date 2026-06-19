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

#include "quadratic.hpp"
#include "refinement_tools.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

std::tuple<float, float, int> quadratic_refinement_method(
    float cost_0,
    float cost_1,
    float cost_2,
    float disp,
    const std::string& measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    (void)disp;
    auto [valid, cost_0_out, cost_1_out, cost_2_out, inv_cost_0, inv_cost_1, inv_cost_2] =
        validate_costs_and_get_variables(cost_0, cost_1, cost_2, measure);

    if (!valid)
        return {0.f, cost_1_out, cst_pandora_msk_pixel_stopped_interpolation};

    // Solve the system: col = alpha * row ** 2 + beta * row + gamma
    // gamma = cost_1
    float alpha = (cost_0_out - 2.f * cost_1_out + cost_2_out) / 2.f;
    float beta = (cost_2_out - cost_0_out) / 2.f;

    // If the costs are close, the result of -b / 2a (minimum) is bounded between [-1, 1]
    // sub_disp is row
    float sub_disp = std::min(1.f, std::max(-1.f, -beta / (2.f * alpha)));

    // sub_cost is col
    float sub_cost = (alpha * sub_disp*sub_disp) + (beta * sub_disp) + cost_1_out;

    return {sub_disp, sub_cost, 0};
}

