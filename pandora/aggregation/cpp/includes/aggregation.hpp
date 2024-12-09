#ifndef CBCA_HPP
#define CBCA_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> cbca_step_1(py::array_t<float> input);

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_2(
    py::array_t<float> step1,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
);

py::array_t<float> cbca_step_3(py::array_t<float> step2);

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_4(
    py::array_t<float> step3,
    py::array_t<float> sum2,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
);

py::array_t<int16_t> cross_support(py::array_t<float> image, int16_t len_arms, float intensity);

#endif  // CBCA_HPP