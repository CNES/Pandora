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
This module contains functions associated to the Image Tools in cpp.
*/

#ifndef IMG_TOOLS_HPP
#define IMG_TOOLS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Find valid neighbors along directions.
 *
 * @param dirs directions in which valid neighbors will be searched, around the (row, col) pixel.
 * Example: [[0,1],[1,0],[0,-1],[-1,0]] searches in the axis-aligned cross centered on the pixel.
 * @param disp disparity map
 * @param valid validity mask
 * @param row current row value
 * @param col current column value
 * @param msk_pixel_invalid value for the PANDORA_MSK_PIXEL_INVALID constant
 * @return valid neighbors
 */
py::array_t<float> find_valid_neighbors(
    py::array_t<float> dirs,
    py::array_t<float> disp,
    py::array_t<int> valid,
    size_t row,
    size_t col,
    int msk_pixel_invalid
);

/**
 * @brief Interpolation of the input image to resolve invalid (nodata) pixels.
 * Interpolates invalid pixels by finding the nearest valid pixels in 8 directions
 * and uses the median of their disparities.
 * Based on: HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
 * IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.
 *
 * @param img input image
 * @param valid validity mask
 * @param msk_pixel_invalid value for the PANDORA_MSK_PIXEL_INVALID constant
 * @param msk_pixel_filled_nodata value for the PANDORA_MSK_PIXEL_FILLED_NODATA constant
 * @return tuple of interpolated input image and updated validity mask
 */
std::tuple<py::array_t<float>, py::array_t<int>> interpolate_nodata_sgm(
    py::array_t<float> img, py::array_t<int> valid,
    int msk_pixel_invalid, int msk_pixel_filled_nodata
);

#endif  // IMG_TOOLS_HPP