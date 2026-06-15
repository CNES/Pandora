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

namespace py = pybind11;

std::tuple<float, float, int> quadratic_refinement_method_impl(
    float c0,
    float c1,
    float c2,
    float disp,
    const std::string& measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    (void)disp;
    auto [valid, c0_out, c1_out, c2_out, ic0, ic1, ic2] =
        validate_costs_and_get_variables(c0, c1, c2, measure);

    if (!valid)
        return {0.f, c1_out, cst_pandora_msk_pixel_stopped_interpolation};

    // Solve the system: col = alpha * row ** 2 + beta * row + gamma
    // gamma = c1
    float alpha = (c0_out - 2.f * c1_out + c2_out) / 2.f;
    float beta = (c2_out - c0_out) / 2.f;

    // If the costs are close, the result of -b / 2a (minimum) is bounded between [-1, 1]
    // sub_disp is row
    float sub_disp = std::min(1.f, std::max(-1.f, -beta / (2.f * alpha)));

    // sub_cost is col
    float sub_cost = (alpha * sub_disp*sub_disp) + (beta * sub_disp) + c1_out;

    return {sub_disp, sub_cost, 0};
}

std::tuple<float, float, int> quadratic_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    auto r_cost = cost.unchecked<1>();
    return quadratic_refinement_method_impl(
        r_cost(0),
        r_cost(1),
        r_cost(2),
        disp,
        measure,
        cst_pandora_msk_pixel_stopped_interpolation
    );
}
