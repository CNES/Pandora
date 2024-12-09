#include <pybind11/pybind11.h>
#include "includes/risk.hpp"
#include "includes/interval_bounds.hpp"
#include "includes/ambiguity.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cost_volume_confidence_cpp, m) {
    m.doc() = "Cost volume confidence functions implemented in C++ with Pybind11";
    
    m.def(
        "compute_ambiguity", 
        &compute_ambiguity, 
        "Computes ambiguity."
    );
    m.def(
        "compute_ambiguity_and_sampled_ambiguity", 
        &compute_ambiguity_and_sampled_ambiguity, 
        "Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks."
    );

    m.def(
        "compute_interval_bounds", 
        &compute_interval_bounds, 
        "Computes interval bounds on the disparity."
    );

    m.def(
        "compute_risk", 
        &compute_risk, 
        "Computes minimum and maximum risk."
    );
    m.def(
        "compute_risk_and_sampled_risk", 
        &compute_risk_and_sampled_risk, 
        "Computes minimum and maximum risk and sampled risk."
    );

}