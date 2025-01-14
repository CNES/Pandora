#include "quadratic.hpp"
#include "refinement_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

std::tuple<float, float, int> quadratic_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    auto [valid, c0, c1, c2, ic0, ic1, ic2] = validate_costs_and_get_variables(cost, measure);

    if (!valid) 
        return {0.f, c1, cst_pandora_msk_pixel_stopped_interpolation};

    float alpha = (c0 - 2.f * c1 + c2) / 2.f;
    float beta = (c2 - c0) / 2.f;

    float sub_disp = std::min(1.f, std::max(-1.f, -beta / (2.f * alpha)));

    float sub_cost = (alpha * sub_disp*sub_disp) + (beta * sub_disp) + c1;

    return {sub_disp, sub_cost, 0};
}