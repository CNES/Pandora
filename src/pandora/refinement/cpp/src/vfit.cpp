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

#include "vfit.hpp"
#include "refinement_tools.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace py = pybind11;

std::tuple<float, float, int> vfit_refinement_method_impl(
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

    // The problem is to approximate sub_cost function with an affine function: y = a * x + origin
    // Calculate the slope
    float a = ic0 > ic2 ? c0_out - c1_out : c2_out - c1_out;

    // Compare the difference disparity between (cost[0]-cost[1]) and (cost[2]-cost[1]):
    // the highest cost is used
    if ( std::abs(a) < 1.0e-15 ) {
        return {0.f, c1_out, 0};
    }

    // Problem is resolved with tangents equality, due to the symmetric V shape of
    // 3 points (cv0, cv2 and (x,y))
    // sub_disp is dx
    float sub_disp = (c0_out - c2_out) / (2 * a);

    // sub_cost is y
    float sub_cost = a * (sub_disp - 1) + c2_out;

    return {sub_disp, sub_cost, 0};
}

std::tuple<float, float, int> vfit_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    auto r_cost = cost.unchecked<1>();
    return vfit_refinement_method_impl(
        r_cost(0),
        r_cost(1),
        r_cost(2),
        disp,
        measure,
        cst_pandora_msk_pixel_stopped_interpolation
    );
}
