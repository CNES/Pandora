#include <pybind11/pybind11.h>
#include "interval_tools.hpp"

namespace py = pybind11;

PYBIND11_MODULE(interval_tools_cpp, m) {
    m.doc() = "Interval tools functions implemented in C++ with Pybind11";

    m.def(
        "create_connected_graph", 
        &create_connected_graph, 
        "Create a boolean connection matrix from segment coordinates"
    );
    m.def(
        "graph_regularization", 
        &graph_regularization, 
        "Regularize the intervals based on quantiles and a given connection graph."
    );

}
