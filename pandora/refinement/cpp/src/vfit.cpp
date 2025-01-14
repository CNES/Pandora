#include "vfit.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

std::tuple<float, float, int> vfit_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
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

    float a = ic0 > ic2 ? c0 - c1 : c2 - c1;

    if ( std::abs(a) < 1.0e-15 ) {
        return {0.f, c1, 0};
    }

    float sub_disp = (c0 - c2) / (2 * a);

    float sub_cost = a * (sub_disp - 1) + c2;

    return {sub_disp, sub_cost, 0};
}