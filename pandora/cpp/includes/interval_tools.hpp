#ifndef INTERVAL_TOOLS_HPP
#define INTERVAL_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<bool> create_connected_graph(
    py::array_t<int> border_left, py::array_t<int> border_right, int depth
);

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<bool>> graph_regularization(
    py::array_t<float> interval_inf,
    py::array_t<float> interval_sup,
    py::array_t<int> border_left,
    py::array_t<int> border_right,
    py::array_t<bool> connection_graph,
    float quantile
);

#endif  // INTERVAL_TOOLS_HPP