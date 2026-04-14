/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

#include "matching_cost.hpp"
#include <cmath>
#include <limits>
#include <iostream>

namespace py = pybind11;

py::array_t<float> reverse_cost_volume(
    py::array_t<float> left_cv,
    int min_disp
) {
    auto r_left_cv = left_cv.unchecked<3>();
    
    size_t n_row = r_left_cv.shape(0);
    size_t n_col = r_left_cv.shape(1);
    size_t n_disp = r_left_cv.shape(2);

    py::array_t<float> right_cv = py::array_t<float>({n_row, n_col, n_disp});
    auto rw_right_cv = right_cv.mutable_unchecked<3>();

    // Fast loop using unchecked access
    for (size_t i = 0; i < n_row; ++i) {
        for (size_t j = 0; j < n_col; ++j) {
            for (size_t d = 0; d < n_disp; ++d) {
                
                size_t col = static_cast<size_t>(static_cast<int>(j + d) + min_disp);
                
                // col is unsigned, so >=n_col does both <0 (cast to basically +inf) and >=n_col
                if (col >= n_col)
                    rw_right_cv(i, j, d) = std::numeric_limits<float>::quiet_NaN();
                else                
                    rw_right_cv(i, j, d) = r_left_cv(i, col, n_disp - 1 - d);
            }
        }
    }

    return right_cv;
}


std::tuple<py::array_t<float>, py::array_t<float>> reverse_disp_range(
    py::array_t<float> left_min,
    py::array_t<float> left_max
) {
    auto r_left_min = left_min.unchecked<2>();
    auto r_left_max = left_max.unchecked<2>();
    
    size_t n_row = r_left_min.shape(0);
    size_t n_col = r_left_min.shape(1);

    py::array_t<float> right_min = py::array_t<float>({n_row, n_col});
    py::array_t<float> right_max = py::array_t<float>({n_row, n_col});
    auto rw_right_min = right_min.mutable_unchecked<2>();
    auto rw_right_max = right_max.mutable_unchecked<2>();

    // init the min and max values at inf
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            rw_right_min(row, col) =  std::numeric_limits<float>::infinity();
            rw_right_max(row, col) = -std::numeric_limits<float>::infinity();
        }
    }

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {

            float d_min_raw = r_left_min(row, col);
            float d_max_raw = r_left_max(row, col);

            // skip nans
            if (std::isnan(d_min_raw))
                continue;
            if (std::isnan(d_max_raw))
                continue;

            int d_min = static_cast<int>(d_min_raw);
            int d_max = static_cast<int>(d_max_raw);
            
            for (int d = d_min; d <= d_max; d++) {
                
                int right_col = static_cast<int>(col) + d;
                
                // increment d when right_col is too low, break when too high
                if (right_col < 0)
                    continue;
                if (right_col >= static_cast<int>(n_col))
                    break;

                // update mins and maxs with -d to reach left_img(row, col) from
                // right_img(row, right_col)
                rw_right_min(row, right_col) = std::min(
                    rw_right_min(row, right_col), static_cast<float>(-d)
                );
                rw_right_max(row, right_col) = std::max(
                    rw_right_max(row, right_col), static_cast<float>(-d)
                );
                
            }

        }
    }

    // set the disp ranges that have not been filled to nan
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            if ( std::isinf(rw_right_min(row, col)) ) {
                rw_right_min(row, col) = std::nanf("");
                rw_right_max(row, col) = std::nanf("");
            }
        }
    }

    return {right_min, right_max};
}

#define PRINT_IF_POS(row, col, message) \
    if (row == 1 && col == 1) { \
        std::cout << message << std::endl; \
    }

#define PRINT_IF_POS_2(row, col, disp, message) \
    if (row == 1 && col == 1 && disp == 2) { \
        std::cout << message << std::endl; \
    }

void cv_masked(
    py::array_t<float> cost_volume,
    py::array_t<float> mask_left,
    py::array_t<float> mask_right_native,
    py::array_t<float> mask_right_shift,
    py::array_t<float> disp_min,
    py::array_t<float> disp_max,
    py::array_t<float> disp_range,
    int min_disp,
    int subpix
) {
    auto rw_cv = cost_volume.mutable_unchecked<3>();
    auto r_mask_left = mask_left.unchecked<2>();
    auto r_mask_right_native = mask_right_native.unchecked<2>();
    auto r_mask_right_shift = mask_right_shift.unchecked<2>();
    auto r_disp_min = disp_min.unchecked<2>();
    auto r_disp_max = disp_max.unchecked<2>();
    auto r_disps = disp_range.unchecked<1>();

    size_t n_row = rw_cv.shape(0);
    size_t n_col = rw_cv.shape(1);
    size_t n_disp = rw_cv.shape(2);
    
    size_t n_col_mask_right_native = r_mask_right_native.shape(1);
    size_t n_col_mask_right_shift = r_mask_right_shift.shape(1);

    float nan_val = std::numeric_limits<float>::quiet_NaN();

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            // If pixel is masked in left image, fill entire disparity range with NaN
            if (std::isnan(r_mask_left(row, col))) {
                for (size_t d = 0; d < n_disp; ++d) {
                    rw_cv(row, col, d) = nan_val;
                }
                continue;
            }

            // Get local disparity range for this pixel
            float local_disp_min = r_disp_min(row, col);
            float local_disp_max = r_disp_max(row, col);

            // Process each disparity
            for (size_t d = 0; d < n_disp; ++d) {
                float disp_value = r_disps(d);

                // Check if disparity is within local range
                if (disp_value < local_disp_min || disp_value > local_disp_max) {
                    rw_cv(row, col, d) = nan_val;
                    continue;
                }

                // Determine which right mask to use
                bool use_shifted = (static_cast<int>(d) % subpix != 0);

                // Calculate right column (image coords)
                float right_col_float = col + disp_value;
                int right_col = static_cast<int>(std::floor(right_col_float));

                float mask_value = nan_val;
                if (right_col >= 0) { // Check left bound

                    // The right bound check depends on whether we use shifted or native mask
                    if (use_shifted && n_col_mask_right_shift > 0 && right_col < static_cast<int>(n_col_mask_right_shift)) {
                        mask_value = r_mask_right_shift(row, right_col);
                    } else if (!use_shifted && right_col < static_cast<int>(n_col_mask_right_native)) {
                        mask_value = r_mask_right_native(row, right_col);
                    } else {
                        // Out of bounds, mask with NaN
                        mask_value = nan_val;
                    }

                } else {
                    mask_value = nan_val;
                }

                // Right pixel is masked
                if (std::isnan(mask_value)) {
                    rw_cv(row, col, d) = nan_val;
                } else if (mask_value != 0.0f) {
                    rw_cv(row, col, d) = nan_val;
                }
            }
        }
    }
}