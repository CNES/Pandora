#ifndef AMBIGUITY_HPP
#define AMBIGUITY_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> compute_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool type_measure_min
);

std::tuple<py::array_t<float>, py::array_t<float>> compute_ambiguity_and_sampled_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range
);

#endif  // AMBIGUITY_HPP