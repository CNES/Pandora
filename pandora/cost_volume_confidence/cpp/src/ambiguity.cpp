#include "ambiguity.hpp"
#include "cost_volume_confidence_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::array_t<float> compute_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool type_measure_min
) {

    auto r_cv = cv.unchecked<3>();
    auto r_grids = grids.unchecked<3>();
    auto r_etas = etas.unchecked<1>();

    size_t n_row = cv.shape(0);
    size_t n_col = cv.shape(1);
    size_t n_disp = cv.shape(2);

    py::array_t<float> ambiguity = py::array_t<float>({n_row, n_col});
    auto rw_amb = ambiguity.mutable_unchecked<2>();

    py::array_t<float> min_img = py::array_t<float>({n_row, n_col});
    py::array_t<float> max_img = py::array_t<float>({n_row, n_col});
    auto rw_min_img = min_img.mutable_unchecked<2>();
    auto rw_max_img = max_img.mutable_unchecked<2>();

    // min and max cost for normalization
    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    float pix_min_cost;
    float pix_max_cost;
    float val;
    bool insert_nan;
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            pix_min_cost = std::numeric_limits<float>::infinity();
            pix_max_cost = -std::numeric_limits<float>::infinity();
            insert_nan = true;
            for (int k = 0; k < n_disp; ++k) {
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

    float extremum_cost = type_measure_min ? min_cost : max_cost;
    float diff_cost = max_cost - min_cost;

    size_t idx_disp_min;
    size_t idx_disp_max;

    float norm_extremum;
    float* normalized_pix_costs = new float[n_disp];
    float cv_val;
    float amb_sum = 0;
    bool amb_status;
    for (int row = 0; row < n_row; ++row) {
        for (int col = 0; col < n_col; ++col) {

            if (type_measure_min) {
                norm_extremum = (rw_min_img(row, col) - extremum_cost) / diff_cost;
            } else {
                norm_extremum = (rw_max_img(row, col) - extremum_cost) / diff_cost;
            }

            if (std::isnan(norm_extremum)) {
                rw_amb(row, col) = nbr_etas * n_disp;
                continue;
            }

            idx_disp_min = searchsorted(disparity_range, r_grids(0, row, col));
            idx_disp_max = searchsorted(disparity_range, r_grids(1, row, col)) + 1;

            // fill normalized cv for this pixel (+-inf when encountering NaNs)
            int nb_minfs = 0;
            int nb_pinfs = 0;
            for (int disp = 0; disp < n_disp; ++disp) {
                cv_val = r_cv(row, col, disp);
                if (std::isnan(cv_val)) {
                    if (disp >= idx_disp_min && disp < idx_disp_max) {
                        normalized_pix_costs[disp] = -std::numeric_limits<float>::infinity();
                        nb_minfs++;
                    }
                    else {
                        normalized_pix_costs[disp] = std::numeric_limits<float>::infinity();
                        nb_pinfs++;
                    } 
                    continue;
                }
                normalized_pix_costs[disp] = (cv_val - extremum_cost) / diff_cost;
            }

            amb_sum = 0;
            if (type_measure_min) {
                for (int eta = 0; eta < nbr_etas; ++eta) {
                    for (int disp = 0; disp < n_disp; ++disp) {
                        amb_status = normalized_pix_costs[disp] <= (norm_extremum + r_etas(eta));
                        amb_sum += amb_status ? 1.f : 0.f;
                    }
                }
            } else {
                for (int eta = 0; eta < nbr_etas; ++eta) {
                    for (int disp = 0; disp < n_disp; ++disp) {
                        amb_status = normalized_pix_costs[disp] >= (norm_extremum - r_etas(eta));
                        amb_sum += amb_status ? 1.f : 0.f;
                    }
                }
            }

            rw_amb(row, col) = amb_sum;

        }
    }

    delete[] normalized_pix_costs;

    return ambiguity;    

}


std::tuple<py::array_t<float>, py::array_t<float>> compute_ambiguity_and_sampled_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range
) {

    auto r_cv = cv.unchecked<3>();
    auto r_grids = grids.unchecked<3>();
    auto r_etas = etas.unchecked<1>();

    size_t n_row = cv.shape(0);
    size_t n_col = cv.shape(1);
    size_t n_disp = cv.shape(2);

    py::array_t<float> ambiguity = py::array_t<float>({n_row, n_col});
    py::array_t<float> sampled_ambiguity = py::array_t<float>({
        n_row,
        n_col,
        static_cast<size_t>(nbr_etas)
    });
    auto rw_amb = ambiguity.mutable_unchecked<2>();
    auto rw_samp_amb = sampled_ambiguity.mutable_unchecked<3>();

    py::array_t<float> min_img = py::array_t<float>({n_row, n_col});
    auto rw_min_img = min_img.mutable_unchecked<2>();

    // min and max cost for normalization
    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    float pix_min_cost;
    float pix_max_cost;
    float val;
    bool insert_nan;
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            pix_min_cost = std::numeric_limits<float>::infinity();
            pix_max_cost = -std::numeric_limits<float>::infinity();
            insert_nan = true;
            for (int k = 0; k < n_disp; ++k) {
                val = r_cv(i,j,k);
                if ( !std::isnan(val) ) {
                    insert_nan = false;
                    pix_min_cost = std::min(pix_min_cost, val);
                    pix_max_cost = std::max(pix_max_cost, val);
                }
            }
            if (insert_nan) {
                rw_min_img(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            rw_min_img(i, j) = pix_min_cost;
            min_cost = std::min(min_cost, pix_min_cost);
            max_cost = std::max(max_cost, pix_max_cost);
        }
    }

    float diff_cost = max_cost - min_cost;

    size_t idx_disp_min;
    size_t idx_disp_max;

    float normalized_extremum;
    float* normalized_pix_costs = new float[n_disp];
    float cv_val;
    float amb_sum = 0;
    float amb_eta_sum = 0;
    bool amb_status;
    for (int row = 0; row < n_row; ++row) {
        for (int col = 0; col < n_col; ++col) {

            normalized_extremum = (rw_min_img(row, col) - min_cost) / diff_cost;

            if (std::isnan(normalized_extremum)) {
                rw_amb(row, col) = nbr_etas * n_disp;
                for (size_t eta = 0; eta < nbr_etas; ++eta) rw_samp_amb(row, col, eta) = n_disp;
                continue;
            }

            idx_disp_min = searchsorted(disparity_range, r_grids(0, row, col));
            idx_disp_max = searchsorted(disparity_range, r_grids(1, row, col)) + 1;

            // fill normalized cv for this pixel (+-inf when encountering NaNs)
            for (int disp = 0; disp < n_disp; ++disp) {
                cv_val = r_cv(row, col, disp);
                if (std::isnan(cv_val)) {
                    if (disp >= idx_disp_min && disp < idx_disp_max) {
                        normalized_pix_costs[disp] = -std::numeric_limits<float>::infinity();
                    }
                    else {
                        normalized_pix_costs[disp] = std::numeric_limits<float>::infinity();
                    } 
                    continue;
                }
                normalized_pix_costs[disp] = (cv_val - min_cost) / diff_cost;
            }

            amb_sum = 0;
            for (int eta = 0; eta < nbr_etas; ++eta) {
                amb_eta_sum = 0;
                for (int disp = 0; disp < n_disp; ++disp) {
                    amb_status = normalized_pix_costs[disp] <= (normalized_extremum + r_etas(eta));
                    amb_eta_sum += amb_status ? 1.f : 0.f;
                }
                amb_sum += amb_eta_sum;
                rw_samp_amb(row, col, eta) = amb_eta_sum;
            }

            rw_amb(row, col) = amb_sum;

        }
    }

    delete[] normalized_pix_costs;

    return std::make_tuple(ambiguity, sampled_ambiguity);    

}