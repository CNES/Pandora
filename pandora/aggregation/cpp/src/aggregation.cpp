#include "aggregation.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::array_t<float> cbca_step_1(py::array_t<float> input) {

    auto r_input = input.unchecked<2>();

    size_t n_col = r_input.shape(0);
    size_t n_row = r_input.shape(1);

    py::array_t<float> output({n_col, n_row + 1});
    auto rw_output = output.mutable_unchecked<2>();

    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < n_row + 1; ++row) {
            rw_output(col, row) = 0.f;        
        }
    }

    float buff_prev_val;

    for (size_t col = 0; col < n_col; ++col) {
        
        buff_prev_val = 0; 
        
        for (size_t row = 0; row < n_row; ++row) {
            
            float curr_val = r_input(col, row);
    
            if (!std::isnan(curr_val)) {
                buff_prev_val = buff_prev_val + curr_val;
            } // python implementation's else is useless with the buffer
    
            rw_output(col, row) = buff_prev_val;
        }
    }

    return output;
}

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_2(
    py::array_t<float> step1,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
) {

    auto r_step1 = step1.unchecked<2>();
    auto r_cross_left = cross_left.unchecked<3>();
    auto r_cross_right = cross_right.unchecked<3>();
    auto r_range_col = range_col.unchecked<1>();
    auto r_range_col_right = range_col_right.unchecked<1>();

    size_t n_col = r_step1.shape(0);
    size_t n_row = r_step1.shape(1) - 1;

    py::array_t<float> step2({n_col, n_row});
    py::array_t<float> sum_step2({n_col, n_row});
    auto rw_step2 = step2.mutable_unchecked<2>();
    auto rw_sum_step2 = sum_step2.mutable_unchecked<2>();

    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < n_row; ++row) {
            rw_step2(col, row) = 0.f;
            rw_sum_step2(col, row) = 0.f;
        }
    }

    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < r_range_col.shape(0); ++row) {

            int64_t left_range_col = r_range_col(row);
            int64_t right_range_col = r_range_col_right(row);

            int16_t left = std::min(
                r_cross_left(col, left_range_col, 0),
                r_cross_right(col, right_range_col, 0)
            );
            int16_t right = std::min(
                r_cross_left(col, left_range_col, 1),
                r_cross_right(col, right_range_col, 1)
            );

            rw_step2(col, left_range_col) = r_step1(col, left_range_col+right) 
                                          - r_step1(col, left_range_col - left - 1);
            rw_sum_step2(col, left_range_col) += left + right;

        }
    }

    return std::make_tuple(step2, sum_step2);
}

py::array_t<float> cbca_step_3(py::array_t<float> step2) {

    auto r_step2 = step2.unchecked<2>();

    size_t n_col = r_step2.shape(0);
    size_t n_row = r_step2.shape(1);

    py::array_t<float> step3({n_col + 1, n_row});
    auto rw_step3 = step3.mutable_unchecked<2>();

    for (size_t row = 0; row < n_row; ++row) {
        rw_step3(0, row) = r_step2(0, row);
        rw_step3(n_col, row) = 0.f;
    }

    for (size_t col = 1; col < n_col; ++col) {
        for (size_t row = 0; row < n_row; ++row) {
            rw_step3(col, row) = rw_step3(col-1, row) + r_step2(col, row);
        }
    }

    return step3;
}

std::tuple<py::array_t<float>, py::array_t<float>> cbca_step_4(
    py::array_t<float> step3,
    py::array_t<float> sum2,
    py::array_t<int16_t> cross_left,
    py::array_t<int16_t> cross_right,
    py::array_t<int64_t> range_col,
    py::array_t<int64_t> range_col_right
) {
    auto r_step3 = step3.unchecked<2>();
    auto r_sum2 = sum2.unchecked<2>();
    auto r_cross_left = cross_left.unchecked<3>();
    auto r_cross_right = cross_right.unchecked<3>();
    auto r_range_col = range_col.unchecked<1>();
    auto r_range_col_right = range_col_right.unchecked<1>();

    size_t n_col = r_step3.shape(0);
    size_t n_row = r_step3.shape(1);
    size_t n_range_col = range_col.shape(0);

    py::array_t<float> step4({n_col - 1, n_row});
    py::array_t<float> sum4({n_col - 1, n_row});
    auto rw_step4 = step4.mutable_unchecked<2>();
    auto rw_sum4 = sum4.mutable_unchecked<2>();

    for (size_t col = 0; col < n_col - 1; ++col) {
        for (size_t row = 0; row < n_row; ++row) {
            rw_sum4(col, row) = r_sum2(col, row);
            rw_step4(col, row) = 0.f;
        }
    }

    for (size_t col = 0; col < n_col - 1; ++col) {
        for (size_t row = 0; row < n_range_col; ++row) {
            
            int64_t left_range_col = r_range_col(row);
            int64_t right_range_col = r_range_col_right(row);
            int16_t top = std::min(
                r_cross_left(col, left_range_col, 2),
                r_cross_right(col, right_range_col, 2)
            );
            int16_t bot = std::min(
                r_cross_left(col, left_range_col, 3),
                r_cross_right(col, right_range_col, 3)
            );

            int step_col = (int)(col) - top - 1;
            if (step_col < 0) step_col = n_col+step_col;

            rw_step4(col, left_range_col) = r_step3(col + bot, left_range_col) 
                                          - r_step3(step_col, left_range_col);
            rw_sum4(col, left_range_col) += top + bot;

            if (top > 0) {
                float sum = 0;
                for (size_t i = 1; i <= top; ++i) sum += r_sum2(col-i, left_range_col);

                rw_sum4(col, left_range_col) += sum;
            }
            if (bot > 0) {
                float sum = 0;
                for (size_t i = 1; i <= bot; ++i) sum += r_sum2(col+i, left_range_col);

                rw_sum4(col, left_range_col) += sum;
            }
        }
    }

    return std::make_tuple(step4, sum4);
}


py::array_t<int16_t> cross_support(py::array_t<float> image, int16_t len_arms, float intensity) {

    py::buffer_info buf_image = image.request();
    size_t n_col = buf_image.shape[0];
    size_t n_row = buf_image.shape[1];
    auto rw_image = image.mutable_unchecked<2>();

    py::array_t<int16_t> cross(py::array::ShapeContainer({n_col, n_row, 4}));
    auto rw_cross = cross.mutable_unchecked<3>();

    auto get_image_value = [&](size_t col, size_t row) -> float {
        return rw_image(col, row);
    };

    auto set_cross_value = [&](size_t col, size_t row, int16_t left, int16_t right, int16_t up, int16_t bot) {
        rw_cross(col, row, 0) = left;
        rw_cross(col, row, 1) = right;
        rw_cross(col, row, 2) = up;
        rw_cross(col, row, 3) = bot;
    };

    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < n_row; ++row) {

            float current_pixel = get_image_value(col, row);

            if (! std::isfinite(current_pixel)) {
                set_cross_value(col, row, 0, 0, 0, 0);
                continue;
            }

            int16_t left_len = 0;
            for (int left = row - 1; left > std::max(static_cast<int>(row - len_arms), -1); --left) {
                if (std::fabs(current_pixel - get_image_value(col, left)) >= intensity) {
                    break;
                }
                left_len++;
            }
            left_len = std::max(left_len, static_cast<int16_t>(1 * (row >= 1 && std::isfinite(get_image_value(col, row - 1)))));

            int16_t right_len = 0;
            for (int right = row + 1; right < std::min(static_cast<int>(row + len_arms), static_cast<int>(n_row)); ++right) {
                if (std::fabs(current_pixel - get_image_value(col, right)) >= intensity) break;
                right_len++;
            }
            right_len = std::max(right_len, static_cast<int16_t>(1 * (row < n_row - 1 && std::isfinite(get_image_value(col, row + 1)))));

            int16_t up_len = 0;
            for (int up = col - 1; up > std::max(static_cast<int>(col - len_arms), -1); --up) {
                if (std::fabs(current_pixel - get_image_value(up, row)) >= intensity) break;
                up_len++;
            }
            up_len = std::max(up_len, static_cast<int16_t>(1 * (col >= 1 && std::isfinite(get_image_value(col - 1, row)))));

            int16_t bot_len = 0;
            for (int bot = col + 1; bot < std::min(static_cast<int>(col + len_arms), static_cast<int>(n_col)); ++bot) {
                if (std::fabs(current_pixel - get_image_value(bot, row)) >= intensity) break;
                bot_len++;
            }
            bot_len = std::max(bot_len, static_cast<int16_t>(1 * (col < n_col - 1 && std::isfinite(get_image_value(col + 1, row)))));

            set_cross_value(col, row, left_len, right_len, up_len, bot_len);

        }
    }

    return cross;
}