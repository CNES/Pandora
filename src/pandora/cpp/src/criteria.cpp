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