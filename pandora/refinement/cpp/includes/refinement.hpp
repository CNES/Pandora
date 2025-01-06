#ifndef REFINEMENT_HPP
#define REFINEMENT_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int64_t>> loop_refinement(
    py::array_t<float> cv,
    py::array_t<float> disp,
    py::array_t<int64_t> mask,
    double d_min,
    double d_max,
    int subpixel,
    std::string measure,
    std::function<
        std::tuple<float, float, int>(py::array_t<float>, float, std::string)
    > &method,
    int64_t cst_pandora_msk_pixel_invalid, 
    int64_t cst_pandora_msk_pixel_stopped_interpolation 
);

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int64_t>> loop_approximate_refinement(
    py::array_t<float> cv,
    py::array_t<float> disp,
    py::array_t<int64_t> mask,
    double d_min,
    double d_max,
    int subpixel,
    std::string measure,
    std::function<
        std::tuple<float, float, int>(py::array_t<float>, float, std::string)
    > &method,
    int64_t cst_pandora_msk_pixel_invalid, 
    int64_t cst_pandora_msk_pixel_stopped_interpolation 
);

#endif  // REFINEMENT_HPP