/* Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA
 *
 *     https://github.com/CNES/Pandora
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This module contains functions associated to the Criteria in cpp.
*/

#ifndef CRITERIA_HPP
#define CRITERIA_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Find pixels whose disparity range contains one or more invalid pixels
 *
 * @param disps disparity bounds for each pixel
 * @param img_mask mask of pixels considered as nan / invalid
 * @return a mask with affected pixels as true, and unaffected pixels as false
 */
py::array_t<bool> partially_missing_variable_ranges(
    py::array_t<float> disps,
    py::array_t<bool> img_mask
);

#endif  // CRITERIA_HPP