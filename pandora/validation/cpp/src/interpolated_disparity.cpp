#include "interpolated_disparity.hpp"
#include <pybind11/embed.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

inline std::array<float, 8> find_valid_neighbors(
    pybind11::detail::unchecked_reference<float, 2> r_disp,
    pybind11::detail::unchecked_reference<int, 2> r_valid,
    size_t row,
    size_t col,
    int msk_pixel_invalid
) {
    size_t n_col = r_disp.shape(0);
    size_t n_row = r_disp.shape(1);
    
    std::array<float, 16> dirs = {
        0,  1,
       -1,  1,
       -1,  0,
       -1, -1,
        0, -1,
        1, -1,
        1,  0,
        1,  1
    };

    size_t max_path_length = std::max(n_row, n_col);

    std::array<float, 8> out;

    for (size_t dir = 0; dir < 8; ++dir) {
        size_t tmp_col = col;
        size_t tmp_row = row;

        for (size_t i = 0; i < max_path_length; ++i) {
            tmp_row += dirs[dir*2];
            tmp_col += dirs[dir*2+1];

            if ( tmp_col < 0 || tmp_col >= n_col || tmp_row < 0 || tmp_row >= n_row ) {
                out[dir] = std::numeric_limits<float>::quiet_NaN();                
                break;
            }

            if ((r_valid(tmp_col, tmp_row) & msk_pixel_invalid) == 0) {
                out[dir] = r_disp(tmp_col, tmp_row);            
                break;
            }
        }
    }

    return out;
} 

inline float get_second_min_val_abs(std::array<float, 8> buf) {
    
    float min = std::numeric_limits<float>::infinity();
    float min_abs = std::numeric_limits<float>::infinity();
    float sec_min = std::numeric_limits<float>::infinity();
    float sec_min_abs = std::numeric_limits<float>::infinity();
    
    float curr_val;
    float curr_val_abs;
    for (size_t i = 0; i < 8; ++i) {
        curr_val = buf[i];
        curr_val_abs = std::abs(curr_val);
        if (curr_val_abs < min_abs) {
            sec_min_abs = min_abs;
            sec_min = min;
            min_abs = curr_val_abs;
            min = curr_val;
        } else if (curr_val_abs < sec_min_abs) {
            sec_min_abs = curr_val_abs;
            sec_min = curr_val;
        }
    }

    return sec_min;
}

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
) {

    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    size_t n_col = r_disp.shape(0);
    size_t n_row = r_disp.shape(1);

    py::array_t<float> out_disp = py::array_t<float>({n_col, n_row});
    py::array_t<int> out_valid = py::array_t<int>({n_col, n_row});
    auto rw_out_disp = out_disp.mutable_unchecked<2>();
    auto rw_out_valid = out_valid.mutable_unchecked<2>();

    std::array<float, 8> valid_neighbors;
    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < n_row; ++row) {
            rw_out_valid(col, row) = r_valid(col, row);
            if (r_valid(col, row) & msk_pixel_occlusion) {
                valid_neighbors = find_valid_neighbors(r_disp, r_valid, row, col, msk_pixel_invalid);
                
                rw_out_disp(col, row) = get_second_min_val_abs(valid_neighbors);
                rw_out_valid(col, row) += msk_pixel_filled_occlusion - msk_pixel_occlusion;
            } else {
                rw_out_disp(col, row) = r_disp(col, row);
            }
        }
    }

    return std::make_tuple(out_disp, out_valid);

}

template <typename T, size_t N>
T compute_median(std::array<T, N> buf) {

    std::vector<T> data;
    for (size_t i = 0; i < buf.size(); ++i) {
        T val = buf[i];
        if (!std::isnan(val)) {
            data.push_back(val);
        }
    }

    size_t size = data.size();
    if (size == 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    std::sort(data.begin(), data.end());

    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.f;
    } else {
        return data[size / 2];
    }
}


std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_sgm(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch, int msk_pixel_filled_mismatch, int msk_pixel_occlusion, int msk_pixel_invalid
) {

    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    int n_col = r_disp.shape(0);
    int n_row = r_disp.shape(1);

    py::array_t<float> out_disp = py::array_t<float>({n_col, n_row});
    py::array_t<int> out_valid = py::array_t<int>({n_col, n_row});
    auto rw_out_disp = out_disp.mutable_unchecked<2>();
    auto rw_out_valid = out_valid.mutable_unchecked<2>();

    std::array<float, 8> valid_neighbors;
    for (int col = 0; col < n_col; ++col) {
        for (int row = 0; row < n_row; ++row) {

            if (r_valid(col, row) & msk_pixel_mismatch) {

                bool found = false;
                for (int i = std::max(0, col-1); i < std::min(n_col - 1, col + 1) + 1; ++i) {
                    for (int j = std::max(0, row-1); j < std::min(n_row - 1, row + 1) + 1; ++j) {
                        if ((r_valid(i, j) & msk_pixel_occlusion) != 0) {
                            found = true;
                            break;
                        }
                    }
                }

                if (found) {
                    rw_out_disp(col, row) = r_disp(col, row);
                    rw_out_valid(col, row) = r_valid(col, row) - msk_pixel_mismatch + msk_pixel_occlusion;
                    continue;
                }

                valid_neighbors = find_valid_neighbors(r_disp, r_valid, row, col, msk_pixel_invalid);
                float median = compute_median(valid_neighbors);

                rw_out_disp(col, row) = median;
                rw_out_valid(col, row) = r_valid(col, row) + msk_pixel_filled_mismatch - msk_pixel_mismatch;
            } else {
                rw_out_disp(col, row) = r_disp(col, row);
                rw_out_valid(col, row) = r_valid(col, row);
            }

        }
    }

    return std::make_tuple(out_disp, out_valid);

}

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_occlusion_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_occlusion, int msk_pixel_filled_occlusion, int msk_pixel_invalid
) {

    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    size_t n_col = r_disp.shape(0);
    size_t n_row = r_disp.shape(1);

    py::array_t<float> out_disp = py::array_t<float>({n_col, n_row});
    py::array_t<int> out_valid = py::array_t<int>({n_col, n_row});
    auto rw_out_disp = out_disp.mutable_unchecked<2>();
    auto rw_out_valid = out_valid.mutable_unchecked<2>();

    bool msk_valid;
    for (size_t col = 0; col < n_col; ++col) {
        for (size_t row = 0; row < n_row; ++row) {

            if (r_valid(col, row) & msk_pixel_occlusion) {
                
                // find first valid pixel to the left
                // msk represents a row, unlike the numba implementation
                size_t msk = row; // default to row
                for (size_t i = row; i <= row; i--) { // relies on overflow
                    if ( (r_valid(col, i) & msk_pixel_invalid) == 0 ) {
                        msk = i;
                        break;
                    }
                }
                if (msk != row) {
                    msk_valid = (r_valid(col, msk) & msk_pixel_invalid) == 0;
                    rw_out_disp(col, row) = r_disp(col, msk);
                    rw_out_valid(col, row) = r_valid(col, row) 
                                            - msk_pixel_occlusion * msk_valid
                                            + msk_pixel_filled_occlusion * msk_valid;
                    continue;
                }

                for (size_t i = row; i < n_row; i++) {
                    if ( (r_valid(col, i) & msk_pixel_invalid) == 0 ) {
                        msk = i;
                        break;
                    }
                }

                msk_valid = (r_valid(col, msk) & msk_pixel_invalid) == 0;
                rw_out_disp(col, row) = r_disp(col, msk);
                rw_out_valid(col, row) = r_valid(col, row) 
                                        - msk_pixel_occlusion * msk_valid
                                        + msk_pixel_filled_occlusion * msk_valid;
            } else {
                rw_out_disp(col, row) = r_disp(col, row);
                rw_out_valid(col, row) = r_valid(col, row);
            }
        }
    }

    return std::make_tuple(out_disp, out_valid);

}

std::tuple<py::array_t<float>, py::array_t<int>> interpolate_mismatch_mc_cnn(
    py::array_t<float> disp, py::array_t<int> valid, 
    int msk_pixel_mismatch, int msk_pixel_filled_mismatch, int msk_pixel_invalid
) {

    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();
    int n_col = r_disp.shape(0);
    int n_row = r_disp.shape(1);

    size_t max_path_length = (size_t)std::max(n_col, n_row);

    py::array_t<float> out_disp = py::array_t<float>({n_col, n_row});
    py::array_t<int> out_valid = py::array_t<int>({n_col, n_row});
    auto rw_out_disp = out_disp.mutable_unchecked<2>();
    auto rw_out_valid = out_valid.mutable_unchecked<2>();

    float dirs[] = {
        0.0, 1.0,
        -0.5, 1.0,
        -1.0, 1.0,
        -1.0, 0.5,
        -1.0, 0.0,
        -1.0, -0.5,
        -1.0, -1.0,
        -0.5, -1.0,
        0.0, -1.0,
        0.5, -1.0,
        1.0, -1.0,
        1.0, -0.5,
        1.0, 0.0,
        1.0, 0.5,
        1.0, 1.0,
        0.5, 1.0
    };

    std::array<float, 16> interp_mismatched;
    for (int col = 0; col < n_col; ++col) {
        for (int row = 0; row < n_row; ++row) {

            if (r_valid(col, row) & msk_pixel_mismatch) {

                int tmp_row;
                int tmp_col;
                for (size_t dir = 0; dir < 16; ++dir) {
                    interp_mismatched[dir] = 0.f;
                    for (size_t i = 0; i < max_path_length; ++i) {
                        tmp_row = std::floor( row + (int)(dirs[2*dir] * i) );
                        tmp_col = std::floor( col + (int)(dirs[2*dir+1] * i) );

                        if (tmp_col < 0 || tmp_col >= n_col || tmp_row < 0 || tmp_row >= n_row) {
                            interp_mismatched[dir] = std::numeric_limits<float>::quiet_NaN();
                            break;
                        }

                        if ((r_valid(tmp_col, tmp_row) & msk_pixel_invalid) == 0) {
                            interp_mismatched[dir] = r_disp(tmp_col, tmp_row);
                            break;
                        }
                    }
                }

                rw_out_disp(col, row) = compute_median(interp_mismatched);
                rw_out_valid(col, row) = r_valid(col, row) + msk_pixel_filled_mismatch - msk_pixel_mismatch;
                
            } else {
                rw_out_disp(col, row) = r_disp(col, row);
                rw_out_valid(col, row) = r_valid(col, row);
            }

        }
    }

    return std::make_tuple(out_disp, out_valid);

}