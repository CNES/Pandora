#ifndef IMG_TOOLS_HPP
#define IMG_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> find_valid_neighbors(
    py::array_t<float> dirs,
    py::array_t<float> disp,
    py::array_t<int> valid,
    size_t row,
    size_t col,
    int msk_pixel_invalid
);

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_nodata_sgm(
    py::array_t<float> img, py::array_t<int> valid, int msk_pixel_invalid, int msk_pixel_filled_nodata
);

#endif  // IMG_TOOLS_HPP