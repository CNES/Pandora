#ifndef RISK_HPP
#define RISK_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::list compute_risk_and_sampled_risk(
    py::array_t<float> cv,
    py::array_t<float> sampled_ambiguity,
    py::array_t<double> etas,
    int nbr_etas,
    py::array_t<int64_t> grids,
    py::array_t<float> disparity_range,
    bool sample_risk
);

#endif  // RISK_HPP