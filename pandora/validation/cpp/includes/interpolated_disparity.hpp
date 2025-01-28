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
This module contains functions associated to the Interpolated Disparity algorithms in cpp.
*/

#ifndef INTERPOLATED_HPP
#define INTERPOLATED_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Interpolates the left disparity map to resolve occlusion conflicts using SGM.
 *
 * @param disp The disparity map (2D array).
 * @param valid Validity mask (2D array).
 * @param msk_pixel_occlusion Value for the PANDORA_MSK_PIXEL_OCCLUSION constant.
 * @param msk_pixel_filled_occlusion Value for the PANDORA_MSK_PIXEL_FILLED_OCCLUSION constant.
 * @param msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @return A tuple containing the interpolated disparity map and updated validity mask.
 */
std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
);

/**
 * @brief Interpolates the left disparity map to resolve mismatch conflicts using SGM.
 *
 * @param disp The disparity map (2D array).
 * @param valid Validity mask (2D array).
 * @param msk_pixel_mismatch Value for the PANDORA_MSK_PIXEL_MISMATCH constant.
 * @param msk_pixel_filled_mismatch Value for the PANDORA_MSK_PIXEL_FILLED_MISMATCH constant.
 * @param msk_pixel_occlusion Value for the PANDORA_MSK_PIXEL_OCCLUSION constant.
 * @param msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @return A tuple containing the interpolated disparity map and updated validity mask.
 */
std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch,
    int msk_pixel_filled_mismatch,
    int msk_pixel_occlusion,
    int msk_pixel_invalid
);

/**
 * @brief Interpolates the left disparity map to resolve occlusion conflicts using MC-CNN.
 *
 * @param disp The disparity map (2D array).
 * @param valid Validity mask (2D array).
 * @param msk_pixel_occlusion Value for the PANDORA_MSK_PIXEL_OCCLUSION constant.
 * @param msk_pixel_filled_occlusion Value for the PANDORA_MSK_PIXEL_FILLED_OCCLUSION constant.
 * @param msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @return A tuple containing the interpolated disparity map and updated validity mask.
 */
std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
);

/**
 * @brief Interpolates the left disparity map to resolve mismatch conflicts using MC-CNN.
 *
 * @param disp The disparity map (2D array).
 * @param valid Validity mask (2D array).
 * @param msk_pixel_mismatch Value for the PANDORA_MSK_PIXEL_MISMATCH constant.
 * @param msk_pixel_filled_mismatch Value for the PANDORA_MSK_PIXEL_FILLED_MISMATCH constant.
 * @param msk_pixel_invalid Value for the PANDORA_MSK_PIXEL_INVALID constant.
 * @return A tuple containing the interpolated disparity map and updated validity mask.
 */
std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch, int msk_pixel_filled_mismatch, int msk_pixel_invalid
);

#endif  // INTERPOLATED_HPP