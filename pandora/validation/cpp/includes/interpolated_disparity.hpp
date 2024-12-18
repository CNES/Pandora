#ifndef INTERPOLATED_HPP
#define INTERPOLATED_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
);

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch, int msk_pixel_filled_mismatch, int msk_pixel_occlusion, int msk_pixel_invalid
);

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
);

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch, int msk_pixel_filled_mismatch, int msk_pixel_invalid
);

#endif  // INTERPOLATED_HPP