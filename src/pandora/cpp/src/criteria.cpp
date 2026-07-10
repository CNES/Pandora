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

#include "criteria.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;

py::array_t<bool> partially_missing_variable_ranges(
    py::array_t<float> disps,
    py::array_t<bool> img_mask
) {
    auto r_disps = disps.unchecked<3>();
    auto r_img_mask = img_mask.unchecked<2>();
    size_t n_row = r_img_mask.shape(0);
    size_t n_col = r_img_mask.shape(1);
    
    py::array_t<bool> missing = py::array_t<bool>(
        {static_cast<int>(n_row), static_cast<int>(n_col)}
    );
    auto rw_missing = missing.mutable_unchecked<2>();

    // First part: identify all the intervals where there is valid data only
    std::vector<std::vector<int>> intervals(n_row);

    for (size_t row = 0; row < n_row; ++row) {

        // start in invalid (masked) data
        bool last_encounter = true;
        
        for (size_t col = 0; col < n_col; ++col) {
            if (r_img_mask(row, col)) {
                if (!last_encounter) {
                    // col is the start of a suite of masked values
                    intervals[row].push_back(col);
                    last_encounter = true;
                }
            } else {
                if (last_encounter) {
                    // col is the start of a suite of valid values
                    intervals[row].push_back(col);
                    last_encounter = false;
                }
            }
        }
        // if false, we ended in a suite of valid values
        // so we have to add an end to this interval
        if (!last_encounter) {
            intervals[row].push_back(n_col);
        }
    }

    // Second part: for each pixel, check if its full disp range is in valid intervals
    for (size_t row = 0; row < n_row; ++row) {

        const auto& valid_intervals = intervals[row];

        for (size_t col = 0; col < n_col; ++col) {

            int col_min = static_cast<int>(r_disps(0, row, col)) + static_cast<int>(col);
            int col_max = static_cast<int>(r_disps(1, row, col)) + static_cast<int>(col);

            // Check if disp range is fully inside any valid interval
            bool found_valid_interval = false;
            
            // this loop could be optimized if the number of intervals is big 
            // (images with lots of small holes)
            // CARS most likely will produce one interval
            for (size_t i = 0; i + 1 < valid_intervals.size(); i += 2) {
                int start = valid_intervals[i];
                int end = valid_intervals[i + 1];

                // disps strictly contained in the interval
                if (start <= col_min && col_max < end) {
                    found_valid_interval = true;
                    break;
                }
            }

            rw_missing(row, col) = !found_valid_interval;
        }
    }

    return missing;
}

void allocate_right_mask_cpp(
    py::array_t<uint16_t> validity_mask,
    py::array_t<bool> bit_1_skip,
    py::array_t<bool> nodata_dil_mask,
    py::array_t<uint8_t> right_mask,
    int disp_min,
    int disp_max,
    int offset,
    uint16_t flag_validity_right,
    uint16_t flag_nodata_right
) {
    // Fast unchecked accessors: validity_mask is written in-place, others are read-only
    auto rw_validity_mask = validity_mask.mutable_unchecked<2>();
    auto r_bit_1_skip = bit_1_skip.unchecked<1>();
    auto r_nodata_dil_mask = nodata_dil_mask.unchecked<2>();
    auto r_right_mask = right_mask.unchecked<2>();

    // Image dimensions and disparity interval size
    size_t n_row = rw_validity_mask.shape(0);
    size_t n_col = rw_validity_mask.shape(1);
    int disp_count = disp_max - disp_min + 1;
    // Rightmost column reachable in the right image (aggregation window border)
    int col_limit = static_cast<int>(n_col) - offset;

    // For each left pixel, check whether its full disparity interval is invalidated
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            // Columns already flagged as outside the right image (bit_1) are skipped
            if (r_bit_1_skip(col)) {
                continue;
            }

            // Counters: how many disparities are invalidated by mask vs nodata
            int cnt_validity = 0;
            int cnt_nodata = 0;

            // Walk the disparity interval in the right image
            for (int dsp = disp_min; dsp <= disp_max; ++dsp) {
                int col_d = static_cast<int>(col) + dsp;

                // Disparity range outside the right image counts as invalid for both criteria
                if (col_d < offset || col_d >= col_limit) {
                    ++cnt_validity;
                    ++cnt_nodata;
                    continue;
                }

                // Early exit: at least one valid right pixel found, no flag will be raised
                if (r_right_mask(row, col_d) == 0 && !r_nodata_dil_mask(row, col_d)) {
                    break;
                }

                // Invalid pixel in the right validity mask
                if (r_right_mask(row, col_d) != 0) {
                    ++cnt_validity;
                }
                // Nodata pixel (dilated) in the right image
                if (r_nodata_dil_mask(row, col_d)) {
                    ++cnt_nodata;
                }
            }

            // Full interval invalidated by the right validity mask
            if (cnt_validity == disp_count) {
                rw_validity_mask(row, col) |= flag_validity_right;
            }
            // Full interval invalidated by nodata in the right image
            if (cnt_nodata == disp_count) {
                rw_validity_mask(row, col) |= flag_nodata_right;
            }
        }
    }
}