#ifndef INTERVAL_BOUNDS_HPP
#define INTERVAL_BOUNDS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<float>> compute_interval_bounds(
    py::array_t<float> cv,
    py::array_t<float> disp_interval,
    float possibility_threshold,
    float type_factor
);

#endif  // INTERVAL_BOUNDS_HPP