#include "refinement_tools.hpp"
#include <cmath>

namespace py = pybind11;

std::tuple<bool, float, float, float, float, float, float> validate_costs_and_get_variables(
    py::array_t<float>& cost,
    const std::string& measure
) {
    auto r_cost = cost.unchecked<1>();
    float c0 = r_cost(0);
    float c1 = r_cost(1);
    float c2 = r_cost(2);
    
    if (std::isnan(c0) || std::isnan(c2)) {
        // Bit 3 = 1: Information: calculations stopped at the pixel step,
        // sub-pixel interpolation did not succeed
        return {false, c0, c1, c2, 0.f, 0.f, 0.f};
    }

    float inverse = 1.f;
    if (measure.compare("max") == 0) { // a.compare(b) = 0 -> a = b
        // Additive inverse : if a < b then -a > -b
        inverse = -1.f;
    }

    float ic0 = inverse * c0;
    float ic1 = inverse * c1;
    float ic2 = inverse * c2;
    // Check if cost[disp] is the minimum cost (or maximum using similarity measure) before fitting
    // If not, interpolation is not applied
    if ( ic1 > ic0 || ic1 > ic2 ) {
        return {false, c0, c1, c2, 0.f, 0.f, 0.f};
    }

    return {true, c0, c1, c2, ic0, ic1, ic2};
}
