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

#include "ambiguity.hpp"
#include "cost_volume_confidence_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;

py::list compute_ambiguity_and_sampled_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool sample_ambiguity
) {

    py::list to_return = py::list();

    auto r_cv = cv.unchecked<3>();
    auto r_grids = grids.unchecked<3>();
    auto r_etas = etas.unchecked<1>();

    size_t n_row = cv.shape(0);
    size_t n_col = cv.shape(1);
    size_t n_disp = cv.shape(2);

    // integral of ambiguity
    py::array_t<float> ambiguity = py::array_t<float>({n_row, n_col});
    auto rw_amb = ambiguity.mutable_unchecked<2>();
    to_return.append(ambiguity);

    py::array_t<float> samp_amb;
    std::unique_ptr<py::detail::unchecked_mutable_reference<float, 3>> rw_samp_amb;
    if (sample_ambiguity) {
        samp_amb = py::array_t<float>({
            n_row,
            n_col,
            static_cast<size_t>(nbr_etas)
        });
        rw_samp_amb = std::make_unique<py::detail::unchecked_mutable_reference<float, 3>>(
            samp_amb.mutable_unchecked<3>()
        );
        to_return.append(samp_amb);
    }

    auto [min_cost, max_cost, rw_min_img, _] = min_max_cost(r_cv, n_row, n_col, n_disp);

    float diff_cost = max_cost - min_cost;

    size_t idx_disp_min;
    size_t idx_disp_max;

    float norm_extremum;
    // Normalized cost volume for one point
    float* normalized_pix_costs = new float[n_disp];
    float cv_val;
    float amb_sum = 0;
    bool amb_status;
    for (size_t row = 0; row < n_row; ++row) {
        for (size_t col = 0; col < n_col; ++col) {

            // Normalized extremum cost for one point
            norm_extremum = (rw_min_img(row, col) - min_cost) / diff_cost;

            // If all costs are at nan, set the maximum value of the ambiguity for this point
            if (std::isnan(norm_extremum)) {
                rw_amb(row, col) = nbr_etas * n_disp;
                if (sample_ambiguity)
                    for (size_t eta = 0; eta < nbr_etas; ++eta)
                        rw_samp_amb->operator()(row, col, eta) = n_disp;
                continue;
            }

            idx_disp_min = searchsorted(disparity_range, r_grids(0, row, col));
            idx_disp_max = searchsorted(disparity_range, r_grids(1, row, col)) + 1;

            // fill normalized cv for this pixel (+-inf when encountering NaNs)
            int nb_minfs = 0;
            int nb_pinfs = 0;
            for (size_t disp = 0; disp < n_disp; ++disp) {
                cv_val = r_cv(row, col, disp);
                if (std::isnan(cv_val)) {
                    // Mask nan to -inf/inf to increase/decrease the value of the ambiguity
                    // if a point contains nan costs
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
                normalized_pix_costs[disp] = (cv_val - min_cost) / diff_cost;
            }

            // fill sampled ambiguity, compute integral
            amb_sum = 0;
            for (int eta = 0; eta < nbr_etas; ++eta) {
                float amb_eta_sum = 0;
                for (int disp = 0; disp < n_disp; ++disp) {
                    amb_status = normalized_pix_costs[disp] <= (norm_extremum + r_etas(eta));
                    amb_eta_sum += amb_status ? 1.f : 0.f;
                }
                amb_sum += amb_eta_sum;
                if (sample_ambiguity)
                    rw_samp_amb->operator()(row, col, eta) = amb_eta_sum;
            }


            // fill integral ambiguity
            rw_amb(row, col) = amb_sum;

        }
    }

    delete[] normalized_pix_costs;

    return to_return;   

}
