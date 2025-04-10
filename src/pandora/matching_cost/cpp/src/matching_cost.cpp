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