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
This module contains functions associated to the Census Matching Cost algorithm in cpp.
*/

#ifndef CENSUS_HPP
#define CENSUS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Computes the matching cost between img_left and imgs_right. Returns the filled
 * cost volume.
 *
 * @param img_left the left image
 * @param imgs_right the right image(s) in a Python list
 * @param cv the empty cost volume
 * @param disps the list of disparities in the cost volume
 * @param census_width the width of the census filter to apply to the images
 * @param census_height the height of the census filter to apply to the images
 * @return the cost volume filled with Census costs
 */
py::array_t<float> compute_matching_costs(
    py::array_t<float> img_left,
    py::list imgs_right,
    py::array_t<float> cv,
    py::array_t<float> disps,
    size_t census_width,
    size_t census_height
);

#endif  // CENSUS_HPP