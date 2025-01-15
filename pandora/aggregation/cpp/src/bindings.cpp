#include <pybind11/pybind11.h>
#include "aggregation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aggregation_cpp, m) {
    m.doc() = "Aggregation functions implemented in C++ with Pybind11";
    
    m.def(
        "cbca", 
        &cbca, 
        ""
    );
    m.def(
        "cross_support", 
        &cross_support, 
        ""
    );

}