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

#include "matching_cost.hpp"
#include <cmath>
#include <limits>

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