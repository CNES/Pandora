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

#include "cost_volume_confidence_tools.hpp"

size_t searchsorted(const py::array_t<float>& array, float value) {
    auto arr = array.unchecked<1>();

    size_t left = 0;
    size_t right = arr.shape(0) - 1;
    size_t mid;

    while (left < right) {
        mid = left + (right - left) / 2;

        if (arr(mid) < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

std::tuple<float, float,
           pybind11::detail::unchecked_mutable_reference<float, 2>,
           pybind11::detail::unchecked_mutable_reference<float, 2>>
min_max_cost(
    py::detail::unchecked_reference<float, 3> r_cv,
    size_t n_row,
    size_t n_col,
    size_t n_disp
){
    py::array_t<float> min_img = py::array_t<float>({n_row, n_col});
    py::array_t<float> max_img = py::array_t<float>({n_row, n_col});
    auto rw_min_img = min_img.mutable_unchecked<2>();
    auto rw_max_img = max_img.mutable_unchecked<2>();

    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    float pix_min_cost;
    float pix_max_cost;
    float val;
    bool insert_nan;
    for (size_t i = 0; i < n_row; ++i) {
        for (size_t j = 0; j < n_col; ++j) {
            pix_min_cost = std::numeric_limits<float>::infinity();
            pix_max_cost = -std::numeric_limits<float>::infinity();
            insert_nan = true;
            for (size_t k = 0; k < n_disp; ++k) {
                val = r_cv(i,j,k);
                if ( !std::isnan(val) ) {
                    insert_nan = false;
                    pix_min_cost = std::min(pix_min_cost, val);
                    pix_max_cost = std::max(pix_max_cost, val);
                }
            }
            if (insert_nan) {
                rw_min_img(i, j) = std::numeric_limits<float>::quiet_NaN();
                rw_max_img(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            rw_min_img(i, j) = pix_min_cost;
            rw_max_img(i, j) = pix_max_cost;
            min_cost = std::min(min_cost, pix_min_cost);
            max_cost = std::max(max_cost, pix_max_cost);

        }
    }
    return std::make_tuple(min_cost, max_cost, rw_min_img, rw_max_img);
}
