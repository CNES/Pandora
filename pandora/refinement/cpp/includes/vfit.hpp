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
This module contains functions associated to the Vfit Refinement algorithms in cpp.
*/

#ifndef VFIT_HPP
#define VFIT_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Returns the subpixel disparity and cost by matching a symmetric V shape 
 * (linear interpolation).
 *
 * @param cost Array of costs for disp - 1, disp, disp + 1.
 * @param disp The current disparity value.
 * @param measure The type of measure used to create the cost volume ("min" or "max").
 * @param cst_pandora_msk_pixel_stopped_interpolation Value for the
 * PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION constant.
 * @return A tuple containing the disparity shift, refined cost, and pixel state.
 */
std::tuple<float, float, int> vfit_refinement_method(
    py::array_t<float> cost, float disp, std::string measure,
    int cst_pandora_msk_pixel_stopped_interpolation
);

#endif  // VFIT_HPP