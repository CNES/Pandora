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

#include "risk.hpp"
#include "cost_volume_confidence_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>

namespace py = pybind11;

py::list compute_risk_and_sampled_risk(
    py::array_t<float> cv,
    py::array_t<float> sampled_ambiguity,
    py::array_t<double> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool sample_risk
) {

    py::list to_return = py::list();

    auto r_cv = cv.unchecked<3>();
    auto r_samp_amb = sampled_ambiguity.unchecked<3>();
    auto r_etas = etas.unchecked<1>();
    auto r_grids = grids.unchecked<3>();

    size_t n_row = cv.shape(0);
    size_t n_col = cv.shape(1);
    size_t n_disp = cv.shape(2);

    // Initialize min and max risk integral
    py::array_t<float> risk_min = py::array_t<float>({n_row, n_col});
    py::array_t<float> risk_max = py::array_t<float>({n_row, n_col});
    auto rw_risk_min = risk_min.mutable_unchecked<2>();
    auto rw_risk_max = risk_max.mutable_unchecked<2>();

    py::array_t<float> risk_disp_inf = py::array_t<float>({n_row, n_col});
    py::array_t<float> risk_disp_sup = py::array_t<float>({n_row, n_col});
    auto rw_risk_disp_inf = risk_disp_inf.mutable_unchecked<2>();
    auto rw_risk_disp_sup = risk_disp_sup.mutable_unchecked<2>();

    py::array_t<float> samp_risk_min;
    py::array_t<float> samp_risk_max;
    std::unique_ptr<py::detail::unchecked_mutable_reference<float, 3>> rw_samp_risk_min;
    std::unique_ptr<py::detail::unchecked_mutable_reference<float, 3>> rw_samp_risk_max;

    if (sample_risk) {
        // Initialize min and max sampled risks
        samp_risk_min = py::array_t<float>({n_row, n_col, static_cast<size_t>(nbr_etas)});
        samp_risk_max = py::array_t<float>({n_row, n_col, static_cast<size_t>(nbr_etas)});
        rw_samp_risk_min = std::make_unique<py::detail::unchecked_mutable_reference<float, 3>>(
            samp_risk_min.mutable_unchecked<3>()
        );
        rw_samp_risk_max = std::make_unique<py::detail::unchecked_mutable_reference<float, 3>>(
            samp_risk_max.mutable_unchecked<3>()
        );
    }

    auto [min_cost, max_cost, rw_min_img, _] = min_max_cost(r_cv, n_row, n_col, n_disp);

    float diff_cost = max_cost - min_cost;

    float* normalized_pix_costs = new float[n_disp];
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {

            // Normalized extremum cost for one point
            float normalized_extremum = (rw_min_img(row, col) - min_cost) / diff_cost;

            // If all costs are at nan, set the risk at nan for this point
            if (std::isnan(normalized_extremum)) {
                rw_risk_min(row, col) = std::numeric_limits<float>::quiet_NaN();
                rw_risk_max(row, col) = std::numeric_limits<float>::quiet_NaN();

                rw_risk_disp_inf(row, col) = std::numeric_limits<float>::quiet_NaN();
                rw_risk_disp_sup(row, col) = std::numeric_limits<float>::quiet_NaN();
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

            // fill normalized cv for this pixel (mask with +-inf when encountering NaNs)
            for (size_t disp = 0; disp < n_disp; ++disp) {
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
            float sum_for_disp_inf = 0;
            float sum_for_disp_sup = 0;
            for (size_t eta = 0; eta < nbr_etas; ++eta) {
                // Obtain min and max disparities for each sample
                float min_disp_idx = std::numeric_limits<float>::infinity();
                float max_disp_idx = -std::numeric_limits<float>::infinity();
                for (size_t disp = 0; disp < n_disp; ++disp) {
                    if (normalized_pix_costs[disp] > (normalized_extremum + r_etas(eta)))
                        continue;
                    // case normalized_pix_costs[disp] < normalized_extremum is not computed:
                    // every case is > normalized_extremum, only - inf is not dealt with
                    // -inf is for nan disparities in disparity range, that should be considered to increase risk
                    // this is equivalent to put the > normalized_extremum condition, with nan put to normalized_extremum
                    // but gain condition check

                    min_disp_idx = std::min(min_disp_idx, static_cast<float>(disp));
                    max_disp_idx = std::max(max_disp_idx, static_cast<float>(disp));
                }

                int min_index = static_cast<int>(min_disp_idx);
                int max_index = static_cast<int>(max_disp_idx);

                float min_disp = std::numeric_limits<float>::infinity();
                float max_disp = -std::numeric_limits<float>::infinity();
                py::buffer_info buf_info = disparity_range.request();
                float* data_ptr = static_cast<float*>(buf_info.ptr);

                min_disp = data_ptr[min_index];
                max_disp = data_ptr[max_index];
                // add sampled max risk to sum
                float eta_max_disp = max_disp_idx - min_disp_idx;
                // add sampled min risk to sum. risk min is defined as ( (1+risk(p,k)) - amb(p,k) )
                float eta_min_disp = 1 + eta_max_disp - r_samp_amb(row, col, eta);

                // add sampled min and max disp to sum
                sum_for_disp_sup += max_disp;
                sum_for_disp_inf += min_disp;

                sum_for_min += eta_min_disp;
                sum_for_max += eta_max_disp;
                if (sample_risk) {
                    // fill sampled min and max risk
                    rw_samp_risk_min->operator()(row, col, eta) = eta_min_disp;
                    rw_samp_risk_max->operator()(row, col, eta) = eta_max_disp;
                }
            }

            // fill min/max risk for this pixel
            rw_risk_min(row, col) = sum_for_min / nbr_etas;
            rw_risk_max(row, col) = sum_for_max / nbr_etas;

            rw_risk_disp_sup(row, col) = sum_for_disp_sup / nbr_etas;
            rw_risk_disp_inf(row, col) = sum_for_disp_inf /nbr_etas;

        }
    }

    delete[] normalized_pix_costs;

    to_return.append( risk_max );
    to_return.append( risk_min );
    to_return.append( risk_disp_sup );
    to_return.append( risk_disp_inf );
    if (sample_risk) {
        to_return.append( samp_risk_max );
        to_return.append( samp_risk_min );
    }
    return to_return;
}