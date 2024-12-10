#ifndef QUADRATIC_HPP
#define QUADRATIC_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::tuple<float, float, int> quadratic_refinement_method(
    py::array_t<float> cost, float disp, std::string measure, int cst_pandora_msk_pixel_stopped_interpolation
);

#endif  // QUADRATIC_HPP