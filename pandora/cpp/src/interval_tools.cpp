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

#include "interval_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;

inline size_t pos_to_id(size_t x, size_t y, size_t width) {
    // faster than y*w + x for create_connected_graph (better accessing order for cache)
    return x*width + y; 
}

py::array_t<bool> create_connected_graph(
    py::array_t<int> border_left, py::array_t<int> border_right, int depth
) {
    // border_left and border_right are already sorted by argwhere
    // we only need to create a connection graph by looking at neighboors from below

    auto r_left = border_left.unchecked<2>();
    auto r_right = border_right.unchecked<2>();

    size_t n_segments = r_left.shape(0);

    py::array_t<bool> aggregated_graph = py::array_t<bool>({n_segments, n_segments});
    auto rw_aggregated_graph = aggregated_graph.mutable_unchecked<2>();

    if (depth == 0) {
        // identity matrix
        for (size_t i = 0; i < n_segments; i++) {
            for (size_t j = 0; j < n_segments; j++) {
                rw_aggregated_graph(i, j) = i==j;
            }
        }
        return aggregated_graph;
    }

    bool* connection_graph = new bool[n_segments*n_segments];
    for (size_t e = 0; e < n_segments*n_segments; e++) connection_graph[e] = false;

    for (size_t i = 0; i < n_segments; i++) {
        int row_i = r_left(i, 0);
        for (size_t k = i+1; k < n_segments; k++) {
            int left_k0 = r_left(k, 0);
            if (left_k0 == row_i) 
                continue;
            if (left_k0 > row_i + 1)
                break;

            if (r_left(k, 1) <= r_right(i, 1) && r_right(k, 1) >= r_left(i, 1)) {
                connection_graph[pos_to_id(i, k, n_segments)] = true;
                connection_graph[pos_to_id(k, i, n_segments)] = true;
            }
        }
    }
    
    bool* list_lines = new bool[n_segments];
    bool* any_new_points = new bool[n_segments];
    for (size_t i = 0; i < n_segments; i++) {
        
        // construct list_lines
        for (size_t l = 0; l < n_segments; l++)
            list_lines[l] = connection_graph[pos_to_id(i, l, n_segments)];

        for (size_t _d = 1; _d < depth; _d++) {
            
            // reset any_new_points
            for (size_t l = 0; l < n_segments; l++) any_new_points[l] = false;

            // construct any_new_points 
            for (size_t l = 0; l < n_segments; l++) {
                if (list_lines[l]) {
                    for (size_t y = 0; y < n_segments; y++) {
                        if (!any_new_points[y])
                        any_new_points[y] = any_new_points[y] || connection_graph[
                            pos_to_id(l, y, n_segments)
                        ];
                    }
                }
            }

            for (size_t j = 0; j < n_segments; j++) {
                list_lines[j] |= any_new_points[j];
            }
        }

        // copy to output
        for (size_t j = 0; j < n_segments; j++) {
            rw_aggregated_graph(i, j) = list_lines[j];
        }
        rw_aggregated_graph(i, i) = true;

    }

    delete[] connection_graph;
    delete[] any_new_points;
    delete[] list_lines;

    return aggregated_graph;
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<bool>> graph_regularization(
    py::array_t<float> interval_inf,
    py::array_t<float> interval_sup,
    py::array_t<int> border_left,
    py::array_t<int> border_right,
    py::array_t<bool> connection_graph,
    float quantile
) {
    auto r_interval_inf = interval_inf.unchecked<2>();
    auto r_interval_sup = interval_sup.unchecked<2>();
    auto r_border_left = border_left.unchecked<2>();
    auto r_border_right = border_right.unchecked<2>();
    auto r_connection_graph = connection_graph.unchecked<2>();
    
    size_t n_row = r_interval_inf.shape(0);
    size_t n_col = r_interval_inf.shape(1);
    size_t n_segments = r_connection_graph.shape(0);

    py::array_t<float> interval_inf_reg = py::array_t<float>({n_row, n_col});
    py::array_t<float> interval_sup_reg = py::array_t<float>({n_row, n_col});
    py::array_t<bool> mask_regularization = py::array_t<bool>({n_row, n_col});
    auto rw_interval_inf_reg = interval_inf_reg.mutable_unchecked<2>();
    auto rw_interval_sup_reg = interval_sup_reg.mutable_unchecked<2>();
    auto rw_mask_regularization = mask_regularization.mutable_unchecked<2>();

    for (size_t row = 0; row < n_row; row++) {
        for (size_t col = 0; col < n_col; col++) {
            rw_interval_inf_reg(row, col) = r_interval_inf(row, col);
            rw_interval_sup_reg(row, col) = r_interval_sup(row, col);
            rw_mask_regularization(row, col) = false;
        }
    }

    float p = 1.f - quantile;

    for (size_t i = 0; i < n_segments; i++) {

        std::vector<std::pair<int, int>> left_coords;
        std::vector<std::pair<int, int>> right_coords;

        for (size_t j = 0; j < n_segments; j++) {
            if (r_connection_graph(i, j)) {
                left_coords.emplace_back(r_border_left(j, 0), r_border_left(j, 1));
                right_coords.emplace_back(r_border_right(j, 0), r_border_right(j, 1));
            }
        }

        std::vector<int> n_pixels;
        n_pixels.push_back(0);
        for (size_t j = 0; j < left_coords.size(); j++) {
            int length = right_coords[j].second - left_coords[j].second + 1;
            n_pixels.push_back(n_pixels.back() + length);
        }

        // Contains the lengths of the segments
        size_t total_pixels = n_pixels.back();
        std::vector<float> agg_inf;
        std::vector<float> agg_sup;

        for (size_t j = 0; j < n_pixels.size()-1; j++) {
            int start = n_pixels[j];
            int end = n_pixels[j + 1];
            int row = left_coords[j].first;
            int col_start = left_coords[j].second;
            int col_end = right_coords[j].second;

            for (int k = 0; k <= (col_end - col_start); k++) {
                float v_inf = r_interval_inf(row, col_start + k);
                float v_sup = r_interval_sup(row, col_start + k);
                if (!std::isnan(v_inf))
                    agg_inf.push_back(v_inf);
                if (!std::isnan(v_sup))
                    agg_sup.push_back(v_sup);
            }
        }

        std::sort(agg_inf.begin(), agg_inf.end());
        std::sort(agg_sup.begin(), agg_sup.end());

        // assume nb_inf == nb_sup
        size_t nb_inf = agg_inf.size() - 1;
        size_t nb_sup = agg_sup.size() - 1;

        float inf_quantile;
        float sup_quantile;
        if (agg_inf.size() > 0) {

            // linear interpolation
            size_t idx_floor_inf = static_cast<size_t>(p * nb_inf);
            float t = p * nb_inf - idx_floor_inf;
            inf_quantile = idx_floor_inf >= nb_inf ?
                agg_inf[idx_floor_inf] :
                agg_inf[idx_floor_inf] * (1.f - t) + agg_inf[idx_floor_inf + 1] * t;

            size_t idx_floor_sup = static_cast<size_t>(quantile * nb_sup);
            t = quantile * nb_sup - idx_floor_sup;
            sup_quantile = idx_floor_sup >= nb_sup ?
                agg_sup[idx_floor_sup] :
                agg_sup[idx_floor_sup] * (1.f - t) + agg_sup[idx_floor_sup + 1] * t;

        } else {
            inf_quantile = std::numeric_limits<float>::quiet_NaN();
            sup_quantile = std::numeric_limits<float>::quiet_NaN();
        }

        int row = r_border_left(i, 0);
        int col_start = r_border_left(i, 1);
        int col_end = r_border_right(i, 1);

        for (int col = col_start; col <= col_end; col++) {
            rw_interval_inf_reg(row, col) = inf_quantile;
            rw_interval_sup_reg(row, col) = sup_quantile;
            rw_mask_regularization(row, col) = true;
        }
    }

    return { interval_inf_reg, interval_sup_reg, mask_regularization };

}