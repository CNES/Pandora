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

#include "img_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;

py::array_t<float> find_valid_neighbors(
    py::array_t<float> dirs,
    py::array_t<float> disp,
    py::array_t<int> valid,
    size_t row,
    size_t col,
    int msk_pixel_invalid
) {
    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    auto r_dirs = dirs.unchecked<2>();
    size_t n_row = r_disp.shape(0);
    size_t n_col = r_disp.shape(1);
    size_t n_dirs = r_dirs.shape(0);

    // Maximum path length
    size_t max_path_length = std::max(n_col, n_row);

    py::array_t<float> out = py::array_t<float>({n_dirs});
    auto rw_out = out.mutable_unchecked<1>();

    for (size_t dir = 0; dir < n_dirs; ++dir) {

        // Find the first valid pixel in the current path
        size_t tmp_row = row;
        size_t tmp_col = col;

        for (size_t i = 0; i < max_path_length; ++i) {
            tmp_row += r_dirs(dir, 1);
            tmp_col += r_dirs(dir, 0);

            // Edge of the image reached: there is no valid pixel in the current path
            if ( tmp_row < 0 || tmp_row >= n_row || tmp_col < 0 || tmp_col >= n_col ) {
                rw_out(dir) = std::numeric_limits<float>::quiet_NaN();                
                break;
            }

            // First valid pixel
            if ((r_valid(tmp_row, tmp_col) & msk_pixel_invalid) == 0) {
                rw_out(dir) = r_disp(tmp_row, tmp_col);            
                break;
            }
        }
    }

    return out;
} 

float compute_median(pybind11::detail::unchecked_reference<float, 1> buf) {

    std::vector<float> data;
    for (size_t i = 0; i < buf.shape(0); ++i) {
        float val = buf(i);
        if (!std::isnan(val)) {
            data.push_back(val);
        }
    }

    size_t size = data.size();
    if (size == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::sort(data.begin(), data.end());

    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.f;
    } else {
        return data[size / 2];
    }
}

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_nodata_sgm(
    py::array_t<float> img,
    py::array_t<int> valid,
    int msk_pixel_invalid,
    int msk_pixel_filled_nodata
) {
    auto r_img = img.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    size_t n_row = r_img.shape(0);
    size_t n_col = r_img.shape(1);

    // 8 directions : [row, y]
    py::array_t<float> dirs = py::array(
        {8, 2},
        std::vector<float>{
            0,  1,
           -1,  1,
           -1,  0,
           -1, -1,
            0, -1,
            1, -1,
            1,  0,
            1,  1
        }.data()
    );

    // Output disparity map and validity mask
    py::array_t<float> out_img = py::array_t<float>({n_row, n_col});
    py::array_t<int> out_valid = py::array_t<int>({n_row, n_col});
    auto rw_out_img = out_img.mutable_unchecked<2>();
    auto rw_out_valid = out_valid.mutable_unchecked<2>();

    py::array_t<float> valid_neighbors;
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            if (r_valid(row, col) & msk_pixel_invalid) {
                valid_neighbors = find_valid_neighbors(
                    dirs, img, valid, row, col, msk_pixel_invalid
                );
                auto r_valid_neighbors = valid_neighbors.unchecked<1>();
                
                // Median of the 8 pixels
                float median = compute_median(r_valid_neighbors);

                // Update the validity mask : Information : filled nodata pixel
                rw_out_img(row, col) = median;
                rw_out_valid(row, col) = msk_pixel_filled_nodata;

            } else {
                rw_out_img(row, col) = r_img(row, col);
                rw_out_valid(row, col) = r_valid(row, col);
            }
        }
    }

    return std::make_tuple(out_img, out_valid);
}
