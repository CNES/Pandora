#include <pybind11/pybind11.h>
#include "aggregation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aggregation_cpp, m) {
    m.doc() = "Aggregation functions implemented in C++ with Pybind11";
    m.def(
        "cbca_step_1", 
        &cbca_step_1, 
        ""
    );
    m.def(
        "cbca_step_2", 
        &cbca_step_2, 
        ""
    );
    m.def(
        "cbca_step_3", 
        &cbca_step_3, 
        ""
    );
    m.def(
        "cbca_step_4",
        &cbca_step_4,
        ""
    );
    m.def(
        "cross_support", 
        &cross_support, 
        ""
    );

}