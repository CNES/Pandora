#include "risk.hpp"
#include "cost_volume_confidence_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::list compute_risk_and_sampled_risk(
    py::array_t<float> cv,
    py::array_t<float> sampled_ambiguity,
    py::array_t<double> etas,
    int nbr_etas,
    py::array_t<int64_t> grids,
    py::array_t<float> disparity_range,
    bool sample_risk
) {

    py::list to_return = py::list();

    auto r_cv = cv.unchecked<3>();
    auto r_samp_amb = sampled_ambiguity.unchecked<3>();
    auto r_etas = etas.unchecked<1>();
    auto r_grids = grids.unchecked<3>();

    int n_row = cv.shape(0);
    int n_col = cv.shape(1);
    int n_disp = cv.shape(2);

    py::array_t<float> risk_min = py::array_t<float>({n_row, n_col});
    py::array_t<float> risk_max = py::array_t<float>({n_row, n_col});
    auto rw_risk_min = risk_min.mutable_unchecked<2>();
    auto rw_risk_max = risk_max.mutable_unchecked<2>();

    py::array_t<float> samp_risk_min;
    py::array_t<float> samp_risk_max;
    std::unique_ptr<py::detail::unchecked_mutable_reference<float, 3>> rw_samp_risk_min;
    std::unique_ptr<py::detail::unchecked_mutable_reference<float, 3>> rw_samp_risk_max;

    if (sample_risk) {
        samp_risk_min = py::array_t<float>({n_row, n_col, nbr_etas});
        samp_risk_max = py::array_t<float>({n_row, n_col, nbr_etas});
        rw_samp_risk_min = std::make_unique<py::detail::unchecked_mutable_reference<float, 3>>(
            samp_risk_min.mutable_unchecked<3>()
        );
        rw_samp_risk_max = std::make_unique<py::detail::unchecked_mutable_reference<float, 3>>(
            samp_risk_max.mutable_unchecked<3>()
        );
    }

    py::array_t<float> min_img = py::array_t<float>({n_row, n_col});
    auto rw_min_img = min_img.mutable_unchecked<2>();

    // min and max cost for normalization
    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    float pix_min_cost;
    float val;
    bool insert_nan;
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            pix_min_cost = std::numeric_limits<float>::infinity();
            insert_nan = true;
            for (int k = 0; k < n_disp; ++k) {
                val = r_cv(i,j,k);
                if ( !std::isnan(val) ) {
                    insert_nan = false;
                    pix_min_cost = std::min(pix_min_cost, val);
                    max_cost = std::max(max_cost, val);
                }
            }
            if (insert_nan) {
                rw_min_img(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            rw_min_img(i, j) = pix_min_cost;
            min_cost = std::min(min_cost, pix_min_cost);
        }
    }

    float diff_cost = max_cost - min_cost;

    float* normalized_pix_costs = new float[n_disp];
    for (int row = 0; row < n_row; ++row) {
        for (int col = 0; col < n_col; ++col) {

            float normalized_extremum = (rw_min_img(row, col) - min_cost) / diff_cost;

            if (std::isnan(normalized_extremum)) {
                rw_risk_min(row, col) = std::numeric_limits<float>::quiet_NaN();
                rw_risk_max(row, col) = std::numeric_limits<float>::quiet_NaN();
                if (!sample_risk) 
                    continue;
                
                for (size_t eta = 0; eta < nbr_etas; ++eta) {
                    rw_samp_risk_min->operator()(
                        row, col, eta
                    ) = std::numeric_limits<float>::quiet_NaN();
                    rw_samp_risk_max->operator()(
                        row, col, eta
                    ) = std::numeric_limits<float>::quiet_NaN();
                }
                continue;
            }

            size_t idx_disp_min = searchsorted(disparity_range, r_grids(0, row, col));
            size_t idx_disp_max = searchsorted(disparity_range, r_grids(1, row, col)) + 1;

            // fill normalized cv for this pixel (+-inf when encountering NaNs)
            for (int disp = 0; disp < n_disp; ++disp) {
                float cv_val = r_cv(row, col, disp);
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

            float sum_for_min = 0;
            float sum_for_max = 0;
            for (int eta = 0; eta < nbr_etas; ++eta) {
                float min_disp = std::numeric_limits<float>::infinity();
                float max_disp = -std::numeric_limits<float>::infinity();
                for (int disp = 0; disp < n_disp; ++disp) {
                    if (normalized_pix_costs[disp] > (normalized_extremum + r_etas(eta)))
                        continue;

                    min_disp = std::min(min_disp, static_cast<float>(disp));
                    max_disp = std::max(max_disp, static_cast<float>(disp));
                }
                float eta_max_disp = max_disp - min_disp;
                float eta_min_disp = 1 + eta_max_disp - r_samp_amb(row, col, eta);
                sum_for_min += eta_min_disp;
                sum_for_max += eta_max_disp;
                if (sample_risk) {
                    rw_samp_risk_min->operator()(row, col, eta) = eta_min_disp;
                    rw_samp_risk_max->operator()(row, col, eta) = eta_max_disp;
                }
            }

            rw_risk_min(row, col) = sum_for_min / nbr_etas;
            rw_risk_max(row, col) = sum_for_max / nbr_etas;

        }
    }

    delete[] normalized_pix_costs;

    to_return.append( risk_max );
    to_return.append( risk_min );
    if (sample_risk) {
        to_return.append( samp_risk_max );
        to_return.append( samp_risk_min );
    }
    return to_return;    
}