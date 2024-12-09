#include <pybind11/pybind11.h>
#include "includes/aggregation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aggregation_cpp, m) {
    m.doc() = "Aggregation functions implemented in C++ with Pybind11";
    m.def(
        "cbca_step_1", 
        &cbca_step_1, 
        "Calculate the horizontal integral image."
    );
    m.def(
        "cbca_step_2", 
        &cbca_step_2, 
        "Calculate the horizontal matching cost for one disparity."
    );
    m.def(
        "cbca_step_3", 
        &cbca_step_3, 
        "Calculate the vertical integral image from horizontal matching cost."
    );
    m.def(
        "cbca_step_4",
        &cbca_step_4,
        "Fully aggregate the matching cost for one disparity."
    );
    m.def(
        "cross_support", 
        &cross_support, 
        "Compute the cross support for an image"
    );

}