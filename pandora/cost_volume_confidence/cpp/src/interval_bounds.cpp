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

#include "interval_bounds.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;

void argsort(float* arr, size_t len, int* indices) {

    for (size_t i = 0; i < len; ++i) {
        indices[i] = i;
    }

    std::sort(indices, &(indices[len]), [&](size_t i, size_t j) {
        if (std::isnan(arr[i]))
            return false;
        if (std::isnan(arr[j]))
            return true;
        return arr[i] < arr[j];
    });

}

std::tuple<py::array_t<float>, py::array_t<float>> compute_interval_bounds(
    py::array_t<float> cv,
    py::array_t<float> disp_interval,
    float possibility_threshold,
    float type_factor
) {
    // IDEA: instead of transforming the cost curve into a possibility, we can compute the cost
    // threshold T to apply directly to the cost curve to obtain the same result
    // This would be a bit more efficient but way less understandable when reading the code
    // T = (1 - possibility_threshold)*(max_cost - min_cost) + min_cost + np.nanmin(cv[row, col, :])
    // cv[row, col, : ] <= T is True when the disparity is possible

    auto r_cv = cv.unchecked<3>();
    auto r_disp_interval = disp_interval.unchecked<1>();

    size_t n_row = r_cv.shape(0);
    size_t n_col = r_cv.shape(1);
    size_t n_disp = r_cv.shape(2);

    // Minimum and maximum of all costs, useful to normalize the cost volume
    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    float cv_val;
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            for (int k = 0; k < n_disp; ++k) {
                cv_val = r_cv(i,j,k);
                if ( !std::isnan(cv_val) ) {
                    min_cost = std::min(min_cost, cv_val);
                    max_cost = std::max(max_cost, cv_val);
                }
            }
        }
    }

    float diff_cost = max_cost - min_cost;

    py::array_t<float> interval_inf = py::array_t<float>({n_row, n_col});
    py::array_t<float> interval_sup = py::array_t<float>({n_row, n_col});
    auto rw_interval_inf = interval_inf.mutable_unchecked<2>();
    auto rw_interval_sup = interval_sup.mutable_unchecked<2>();

    float* norm_pix_costs = new float[n_disp];
    int* sorted_indices = new int[n_disp];
    float max_pix_cost;
    int min_valid_idx = -1;
    int max_valid_idx = -1;
    bool found;
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {

            max_pix_cost = -std::numeric_limits<float>::infinity();
            
            // Normalized cost volume
            for (int disp = 0; disp < n_disp; ++disp) {
                cv_val = r_cv(row, col, disp);
                norm_pix_costs[disp] = (cv_val - min_cost) / diff_cost;
                if (!std::isnan(cv_val)) {
                    max_pix_cost = std::max(max_pix_cost, type_factor*norm_pix_costs[disp]);
                }
            }

            // possibility transformation
            for (int disp = 0; disp < n_disp; ++disp) {
                if (!std::isnan(norm_pix_costs[disp]))
                    norm_pix_costs[disp] = type_factor * norm_pix_costs[disp] + 1.f - max_pix_cost;
            }

            // Computing the interval bounds by applying a threshold to the possibility distribution
            argsort(norm_pix_costs, n_disp, sorted_indices);

            found = false;
            for (int disp = 0; disp < n_disp; ++disp) {
                if (norm_pix_costs[disp] >= possibility_threshold) {
                    found = true;
                    break;
                }
            }

            // If the cost curve is all NaN, put NaN for the moment
            // This allows to not take them into account during regularization
            if (!found) {
                rw_interval_inf(row, col) = std::numeric_limits<float>::quiet_NaN();
                rw_interval_sup(row, col) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            min_valid_idx = std::numeric_limits<float>::infinity();
            max_valid_idx = -std::numeric_limits<float>::infinity();
            for (int disp = 0; disp < n_disp; ++disp) {
                if (norm_pix_costs[sorted_indices[disp]] >= possibility_threshold) {
                    min_valid_idx = std::min(min_valid_idx, sorted_indices[disp]);
                    max_valid_idx = std::max(max_valid_idx, sorted_indices[disp]);
                }
            }

            // If the interval bounds are the minima of the cost curve,
            // extending the interval (+/- 1) because of the disparity refinement
            if (min_valid_idx > 0 && static_cast<int>(norm_pix_costs[min_valid_idx])==1)
                --min_valid_idx;
            if (max_valid_idx < n_disp-1 && static_cast<int>(norm_pix_costs[max_valid_idx])==1)
                ++max_valid_idx;

            rw_interval_inf(row, col) = r_disp_interval(min_valid_idx);
            rw_interval_sup(row, col) = r_disp_interval(max_valid_idx);
        }
    }

    delete[] norm_pix_costs;
    delete[] sorted_indices;

    return std::make_tuple(interval_inf, interval_sup);

}