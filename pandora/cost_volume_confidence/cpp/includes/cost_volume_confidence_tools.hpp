#ifndef COST_VOLUME_CONFIDENCE_TOOLS_HPP
#define COST_VOLUME_CONFIDENCE_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

size_t searchsorted(const py::array_t<float>& array, float value);

#endif  // COST_VOLUME_CONFIDENCE_TOOLS_HPP