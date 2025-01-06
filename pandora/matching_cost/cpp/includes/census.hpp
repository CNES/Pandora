#ifndef CENSUS_HPP
#define CENSUS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> compute_matching_costs(
    py::array_t<float> img_left,
    py::list imgs_right,
    py::array_t<float> cv,
    py::array_t<float> disps,
    size_t census_width,
    size_t census_height
);

#endif  // CENSUS_HPP