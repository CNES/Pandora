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
This module contains functions associated to the Refinement algorithms in cpp.
*/

#ifndef REFINEMENT_HPP
#define REFINEMENT_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;

/**
 * @brief Applies the refinement method for each pixel.
 *
 * @param cv Cost volume to refine.
 * @param disp Disparity map.
 * @param mask Validity mask.
 * @param d_min Minimal disparity.
 * @param d_max Maximal disparity.
 * @param subpixel Subpixel precision used to create the cost volume (1, 2, or 4).
 * @param measure The measure used to create the cost volume.
 * @param method The refinement method.
 * @param cst_pandora_msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @param cst_pandora_msk_pixel_stopped_interpolation Value for the 
 * PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION constant.
 * @return A tuple containing the refine coefficient, refined disparity map, and updated 
 * validity mask.
 */
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

/**
 * @brief Applies the refinement method on the right disparity map created with the approximate
 * method.
 *
 * The approximate method performs a diagonal search for the minimum on the left cost volume.
 *
 * @param cv Left cost volume.
 * @param disp Right disparity map.
 * @param mask Right validity mask.
 * @param d_min Minimal disparity.
 * @param d_max Maximal disparity.
 * @param subpixel Subpixel precision used to create the cost volume (1, 2, or 4).
 * @param measure The type of measure used to create the cost volume ("min" or "max").
 * @param method The refinement method.
 * @param cst_pandora_msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @param cst_pandora_msk_pixel_stopped_interpolation Value for the 
 * PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION constant.
 * @return A tuple containing the refine coefficient, refined disparity map, and updated 
 * validity mask.
 */
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int64_t>> 
loop_approximate_refinement(
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