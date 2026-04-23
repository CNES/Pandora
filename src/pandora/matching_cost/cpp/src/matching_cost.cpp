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
#include <algorithm>
#include <vector>

namespace py = pybind11;

/**
 * @brief Create the right_cv from the left_one by reindexing (i,j,d) -> (i, j + d, -d)
 *
 * @param left_cv: the 3D cost_colume data array, with dimensions row, col, disp
 * @param disp_min: the minimum of the right disparities
 * @return: The right cost volume data
 */
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


/**
 * @brief Create the right disp ranges from the left disp ranges
 *
 * @param left_min: the 2D left disp min array, with dimensions row, col
 * @param left_max: the 2D left disp max array, with dimensions row, col
 * @return: The min and max disp ranges for the right image
 */
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


/**
 * @brief Apply masking to cost volume based on valid pixels and local disparity ranges
 *
 * Masks the cost volume by:
 * - Setting entire disparity range to NaN for pixels masked in left image
 * - Setting disparities outside local [disp_min, disp_max] range to NaN
 * - Setting disparities to NaN if corresponding right pixel is masked
 *
 * @param cost_volume: the 3D cost volume data array (row, col, disp)
 * @param mask_left: the 2D left mask array (row, col)
 * @param mask_right_native: the 2D right mask for whole pixel disparities (row, col)
 * @param mask_right_shift: the 2D right mask for subpix disparities (row, col)
 * @param disp_min: the 2D local minimum disparities (row, col)
 * @param disp_max: the 2D local maximum disparities (row, col)
 * @param disp_range: the 1D disparity range values (disp)
 * @param global_disp_min: global cost volume minimum disparity
 * @param subpix: subpixel precision
 * @return: None
 */
void cv_masked(
    py::array_t<float> cost_volume,
    py::array_t<float> mask_left,
    py::array_t<float> mask_right_native,
    py::array_t<float> mask_right_shift,
    py::array_t<float> disp_min,
    py::array_t<float> disp_max,
    py::array_t<float> disp_range,
    int global_disp_min,
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
    
    int n_col_mask_right_native = static_cast<int>(r_mask_right_native.shape(1));
    int n_col_mask_right_shift = static_cast<int>(r_mask_right_shift.shape(1));

    float nan_val = std::numeric_limits<float>::quiet_NaN();
    float global_disp_min_f = static_cast<float>(global_disp_min);

    // Precompute disparity data :
    // - use shifted or native mask
    // - floor(disp) so we don't recompute it every time
    std::vector<bool> use_shifted(n_disp);
    std::vector<int> disp_floor(n_disp);
    for (size_t d = 0; d < n_disp; ++d) {
        use_shifted[d] = (static_cast<int>(d) % subpix != 0);
        disp_floor[d] = static_cast<int>(std::floor(r_disps(d)));
    }

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            // If pixel is masked in left image, fill entire disparity range with NaN
            if (std::isnan(r_mask_left(row, col))) {
                for (size_t d = 0; d < n_disp; ++d) {
                    rw_cv(row, col, d) = nan_val;
                }
                continue;
            }

            int col_i = static_cast<int>(col);

            // Get local disparity range
            float local_disp_min = r_disp_min(row, col);
            float local_disp_max = r_disp_max(row, col);

            // Get corresponding indices
            int local_min_idx = static_cast<int>(std::ceil((local_disp_min - global_disp_min_f) * subpix));
            int local_max_idx = static_cast<int>(std::floor((local_disp_max - global_disp_min_f) * subpix));

            int valid_start = std::max(local_min_idx, 0);
            int valid_end = std::min(local_max_idx, static_cast<int>(n_disp) - 1);

            // Mask disparities below/above the range
            for (int d = 0; d < valid_start; ++d) {
                rw_cv(row, col, d) = nan_val;
            }
            for (int d = valid_end + 1; d < static_cast<int>(n_disp); ++d) {
                rw_cv(row, col, d) = nan_val;
            }

            if (valid_start > valid_end) {
                // The whole range is invalid (= all disps are already masked)
                continue;
            }

            // Process disps in range
            for (int d = valid_start; d <= valid_end; ++d) {
                int right_col = col_i + disp_floor[d];

                float mask_value = nan_val;
                if (right_col >= 0) { // Check left bound

                    // The right bound check depends on whether we use shifted or native mask
                    if (use_shifted[d] && right_col < n_col_mask_right_shift) {
                        mask_value = r_mask_right_shift(row, right_col);
                    } else if (!use_shifted[d] && right_col < n_col_mask_right_native) {
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