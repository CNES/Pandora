#include "includes/quadratic.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

std::tuple<float, float, int> quadratic_refinement_method(
    py::array_t<float> cost, float disp, std::string measure, int cst_pandora_msk_pixel_stopped_interpolation
) {
    auto r_cost = cost.unchecked<1>();
    float c0 = r_cost(0);
    float c1 = r_cost(1);
    float c2 = r_cost(2);
    
    if (std::isnan(c0) || std::isnan(c2)) {
        return {0.f, c1, cst_pandora_msk_pixel_stopped_interpolation};
    }

    float inverse = 1.f;
    if (measure.compare("max") == 0) { // a.compare(b) = 0 -> a = b
        inverse = -1.f;
    }

    float ic0 = inverse * c0;
    float ic1 = inverse * c1;
    float ic2 = inverse * c2;
    if ( ic1 > ic0 || ic1 > ic2 ) {
        return {0.f, c1, cst_pandora_msk_pixel_stopped_interpolation};
    }

    float alpha = (c0 - 2.f * c1 + c2) / 2.f;
    float beta = (c2 - c0) / 2.f;

    float sub_disp = std::min(1.f, std::max(-1.f, -beta / (2.f * alpha)));

    float sub_cost = (alpha * sub_disp*sub_disp) + (beta * sub_disp) + c1;

    return {sub_disp, sub_cost, 0};
}