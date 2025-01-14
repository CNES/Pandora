#include "vfit.hpp"
#include "refinement_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

std::tuple<float, float, int> vfit_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
) {
    auto [valid, c0, c1, c2, ic0, ic1, ic2] = validate_costs_and_get_variables(cost, measure);

    if (!valid) 
        return {0.f, c1, cst_pandora_msk_pixel_stopped_interpolation};
    
    float a = ic0 > ic2 ? c0 - c1 : c2 - c1;

    if ( std::abs(a) < 1.0e-15 ) {
        return {0.f, c1, 0};
    }

    float sub_disp = (c0 - c2) / (2 * a);

    float sub_cost = a * (sub_disp - 1) + c2;

    return {sub_disp, sub_cost, 0};
}