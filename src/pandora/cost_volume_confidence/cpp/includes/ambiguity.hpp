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
This module contains functions associated to the Ambiguity algorithms in cpp.
*/

#ifndef AMBIGUITY_HPP
#define AMBIGUITY_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Computes ambiguity and sampled ambiguity if needed,
 * useful for evaluating ambiguity in notebooks.
 *
 * @param cv cost volume
 * @param etas range between eta_min and eta_max with step eta_step
 * @param nbr_etas number of etas
 * @param grids array containing min and max disparity grids
 * @param disparity_range array containing disparity range
 * @param sample_ambiguity whether to return the sampled ambiguity along with the ambiguity
 * @return the normalized ambiguity and sampled ambiguity
 */
py::list compute_ambiguity_and_sampled_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool sample_ambiguity
);

#endif  // AMBIGUITY_HPP