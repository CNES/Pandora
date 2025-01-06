#include <pybind11/pybind11.h>
#include "census.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matching_cost_cpp, m) {
    m.doc() = "Matching cost functions implemented in C++ with Pybind11";
    
    m.def(
        "compute_matching_costs", 
        &compute_matching_costs, 
        "Computes matching costs of images."
    );

}