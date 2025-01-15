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
    
    // The problem is to approximate sub_cost function with an affine function: y = a * x + origin
    // Calculate the slope
    float a = ic0 > ic2 ? c0 - c1 : c2 - c1;

    // Compare the difference disparity between (cost[0]-cost[1]) and (cost[2]-cost[1]):
    // the highest cost is used
    if ( std::abs(a) < 1.0e-15 ) {
        return {0.f, c1, 0};
    }

    // Problem is resolved with tangents equality, due to the symmetric V shape of
    // 3 points (cv0, cv2 and (x,y))
    // sub_disp is dx
    float sub_disp = (c0 - c2) / (2 * a);

    // sub_cost is y
    float sub_cost = a * (sub_disp - 1) + c2;

    return {sub_disp, sub_cost, 0};
}