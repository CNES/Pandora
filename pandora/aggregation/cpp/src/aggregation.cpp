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

#include "aggregation.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::array_t<float> cbca_step_1(py::array_t<float> input) {

    auto r_input = input.unchecked<2>();

    size_t n_row = r_input.shape(0);
    size_t n_col = r_input.shape(1);

    // Allocate the intermediate cost volume S_h
    // added a column to manage the case in the step 2 : col - left_arm_length -1 = -1
    py::array_t<float> output({n_row, n_col + 1});
    auto rw_output = output.mutable_unchecked<2>();

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col + 1; ++col) {
            rw_output(row, col) = 0.f;        
        }
    }

    float buff_prev_val;

    for (size_t row = 0; row < n_row; ++row) {
        
        buff_prev_val = 0.f; 
        
        for (size_t col = 0; col < n_col; ++col) {
            
            float curr_val = r_input(row, col);
    
            // Do not propagate nan
            if (!std::isnan(curr_val)) {
                buff_prev_val = buff_prev_val + curr_val;
            } // python implementation's else is useless with the buffer
    
            rw_output(row, col) = buff_prev_val;
        }
    }

    return output;
}

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_2(
    py::array_t<float> step1,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_row,
    py::array_t<int64_t> range_row_right
) {

    auto r_step1 = step1.unchecked<2>();
    auto r_cross_left = cross_left.unchecked<3>();
    auto r_cross_right = cross_right.unchecked<3>();
    auto r_range_row = range_row.unchecked<1>();
    auto r_range_row_right = range_row_right.unchecked<1>();

    size_t n_row = r_step1.shape(0);
    size_t n_col = r_step1.shape(1) - 1;

    // Allocate the intermediate cost volume E_h, remove the extra column from the step 1
    py::array_t<float> step2({n_row, n_col});
    py::array_t<float> sum_step2({n_row, n_col});
    auto rw_step2 = step2.mutable_unchecked<2>();
    auto rw_sum_step2 = sum_step2.mutable_unchecked<2>();

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            rw_step2(row, col) = 0.f;
            rw_sum_step2(row, col) = 0.f;
        }
    }
    
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < r_range_row.shape(0); ++col) {

            int64_t left_range_row = r_range_row(col);
            int64_t right_range_row = r_range_row_right(col);

            int16_t left = std::min(
                r_cross_left(row, left_range_row, 0),
                r_cross_right(row, right_range_row, 0)
            );
            int16_t right = std::min(
                r_cross_left(row, left_range_row, 1),
                r_cross_right(row, right_range_row, 1)
            );

            rw_step2(row, left_range_row) = r_step1(row, left_range_row+right) 
                                          - r_step1(row, left_range_row - left - 1);
            rw_sum_step2(row, left_range_row) += left + right;

        }
    }

    return std::make_tuple(step2, sum_step2);
}

py::array_t<float> cbca_step_3(py::array_t<float> step2) {

    auto r_step2 = step2.unchecked<2>();

    size_t n_row = r_step2.shape(0);
    size_t n_col = r_step2.shape(1);

    // Allocate the intermediate cost volume S_v
    // added a row to manage the case in the step 4 : row - up_arm_length -1 = -1
    py::array_t<float> step3({n_row + 1, n_col});
    auto rw_step3 = step3.mutable_unchecked<2>();

    for (size_t col = 0; col < n_col; ++col) {
        rw_step3(0, col) = r_step2(0, col);
        rw_step3(n_row, col) = 0.f;
    }

    for (size_t row = 1; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            rw_step3(row, col) = rw_step3(row-1, col) + r_step2(row, col);
        }
    }

    return step3;
}

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_4(
    py::array_t<float> step3,
    py::array_t<float> sum2,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_row,
    py::array_t<int64_t> range_row_right
) {
    auto r_step3 = step3.unchecked<2>();
    auto r_sum2 = sum2.unchecked<2>();
    auto r_cross_left = cross_left.unchecked<3>();
    auto r_cross_right = cross_right.unchecked<3>();
    auto r_range_row = range_row.unchecked<1>();
    auto r_range_row_right = range_row_right.unchecked<1>();

    size_t n_row = r_step3.shape(0);
    size_t n_col = r_step3.shape(1);
    size_t n_range_row = range_row.shape(0);

    // Allocate the final cost volume E, remove the extra row from the step 3
    py::array_t<float> step4({n_row - 1, n_col});
    py::array_t<float> sum4({n_row - 1, n_col});
    auto rw_step4 = step4.mutable_unchecked<2>();
    auto rw_sum4 = sum4.mutable_unchecked<2>();

    for (size_t row = 0; row < n_row - 1; ++row) {
        for (size_t col = 0; col < n_col; ++col) {
            rw_sum4(row, col) = r_sum2(row, col);
            rw_step4(row, col) = 0.f;
        }
    }

    for (size_t row = 0; row < n_row - 1; ++row) {
        for (size_t col = 0; col < n_range_row; ++col) {
            
            int64_t left_range_row = r_range_row(col);
            int64_t right_range_row = r_range_row_right(col);
            int16_t top = std::min(
                r_cross_left(row, left_range_row, 2),
                r_cross_right(row, right_range_row, 2)
            );
            int16_t bot = std::min(
                r_cross_left(row, left_range_row, 3),
                r_cross_right(row, right_range_row, 3)
            );

            int step_row = static_cast<int>(row) - top - 1;
            if (step_row < 0) 
                step_row += n_row;

            rw_step4(row, left_range_row) = r_step3(row + bot, left_range_row) 
                                          - r_step3(step_row, left_range_row);
            rw_sum4(row, left_range_row) += top + bot;

            if (top > 0) {
                float sum = 0;
                for (size_t i = 1; i <= top; ++i) sum += r_sum2(row-i, left_range_row);

                rw_sum4(row, left_range_row) += sum;
            }
            if (bot > 0) {
                float sum = 0;
                for (size_t i = 1; i <= bot; ++i) sum += r_sum2(row+i, left_range_row);

                rw_sum4(row, left_range_row) += sum;
            }
        }
    }

    return std::make_tuple(step4, sum4);
}


py::array_t<int16_t> cross_support(py::array_t<float> image, int16_t len_arms, float intensity) {

    py::buffer_info buf_image = image.request();
    size_t n_row = buf_image.shape[0];
    size_t n_col = buf_image.shape[1];
    auto rw_image = image.mutable_unchecked<2>();

    py::array_t<int16_t> cross(py::array::ShapeContainer({n_row, n_col, 4}));
    auto rw_cross = cross.mutable_unchecked<3>();

    auto set_cross_value = [&](
        size_t row, size_t col,
        int16_t left, int16_t right, int16_t up, int16_t bot
    ) {
        rw_cross(row, col, 0) = left;
        rw_cross(row, col, 1) = right;
        rw_cross(row, col, 2) = up;
        rw_cross(row, col, 3) = bot;
    };

    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {

            float current_pixel = rw_image(row, col);

            // If the pixel is valid (np.isfinite = True) compute the cross support
            // Else (np.isfinite = False) the pixel is not valid (no data or invalid) and 
            // the cross support value is 0 for the 4 arms (default value of the variable cross).
            if (! std::isfinite(current_pixel)) {
                set_cross_value(row, col, 0, 0, 0, 0);
                continue;
            }

            int16_t left_len = 0;
            for (int left = col - 1; left > std::max(static_cast<int>(col - len_arms), -1); --left){
                if (std::fabs(current_pixel - rw_image(row, left)) >= intensity)
                    break;
                left_len++;
            }
            // enforces a minimum support region of 3×3 if pixels are valid
            left_len = std::max(
                left_len, 
                static_cast<int16_t>(col >= 1 && std::isfinite(rw_image(row, col - 1)))
            );

            int16_t right_len = 0;
            for (
                int right = col + 1;
                right < std::min(static_cast<int>(col + len_arms), static_cast<int>(n_col));
                ++right
            ) {
                if (std::fabs(current_pixel - rw_image(row, right)) >= intensity)
                    break;
                right_len++;
            }
            // enforces a minimum support region of 3×3 if pixels are valid
            right_len = std::max(
                right_len,
                static_cast<int16_t>(col < n_col - 1 && std::isfinite(rw_image(row, col + 1)))
            );

            int16_t up_len = 0;
            for (int up = row - 1; up > std::max(static_cast<int>(row - len_arms), -1); --up) {
                if (std::fabs(current_pixel - rw_image(up, col)) >= intensity)
                    break;
                up_len++;
            }
            // enforces a minimum support region of 3×3 if pixels are valid
            up_len = std::max(
                up_len, 
                static_cast<int16_t>(row >= 1 && std::isfinite(rw_image(row - 1, col)))
            );

            int16_t bot_len = 0;
            for (
                int bot = row + 1;
                bot < std::min(static_cast<int>(row + len_arms), static_cast<int>(n_row));
                ++bot
            ) {
                if (std::fabs(current_pixel - rw_image(bot, col)) >= intensity) 
                    break;
                bot_len++;
            }
            // enforces a minimum support region of 3×3 if pixels are valid
            bot_len = std::max(
                bot_len,
                static_cast<int16_t>(row < n_row - 1 && std::isfinite(rw_image(row + 1, col)))
            );

            set_cross_value(row, col, left_len, right_len, up_len, bot_len);

        }
    }

    return cross;
}

std::tuple<py::array_t<float>, py::array_t<float>> cbca(
    py::array_t<float> input,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_row,
    py::array_t<int64_t> range_row_right
) {

    // Step 1 : horizontal integral image
    auto step1 = cbca_step_1(input);

    // Step 2 : horizontal matching cost
    auto [step2, sum2] = cbca_step_2(
        step1,
        cross_left,
        cross_right,
        range_row,
        range_row_right
    );

    // Step 3 : vertical integral image
    auto step3 = cbca_step_3(step2);

    // Step 4 : aggregate cost volume
    return cbca_step_4(
        step3,
        sum2,
        cross_left,
        cross_right,
        range_row,
        range_row_right
    );

}