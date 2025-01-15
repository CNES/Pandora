#include "ambiguity.hpp"
#include "cost_volume_confidence_tools.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::list compute_ambiguity_and_sampled_ambiguity(
    py::array_t<float> cv,
    py::array_t<float> etas,
    int nbr_etas,
    py::array_t<int> grids,
    py::array_t<float> disparity_range,
    bool type_measure_min,
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

    py::array_t<float> min_img = py::array_t<float>({n_row, n_col});
    py::array_t<float> max_img = py::array_t<float>({n_row, n_col});
    auto rw_min_img = min_img.mutable_unchecked<2>();
    auto rw_max_img = max_img.mutable_unchecked<2>();

    // Minimum and maximum of all costs, useful to normalize the cost volume
    float min_cost = std::numeric_limits<float>::infinity();
    float max_cost = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            float pix_min_cost = std::numeric_limits<float>::infinity();
            float pix_max_cost = -std::numeric_limits<float>::infinity();
            bool insert_nan = true;
            for (int k = 0; k < n_disp; ++k) {
                float val = r_cv(i,j,k);
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
    // Normalized cost volume for one point
    float* normalized_pix_costs = new float[n_disp];
    float cv_val;
    float amb_sum = 0;
    bool amb_status;
    for (int row = 0; row < n_row; ++row) {
        for (int col = 0; col < n_col; ++col) {

            // Normalized extremum cost for one point
            norm_extremum = type_measure_min*(rw_min_img(row, col) - extremum_cost) / diff_cost 
                            + !type_measure_min*(rw_max_img(row, col) - extremum_cost) / diff_cost;

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
            for (int disp = 0; disp < n_disp; ++disp) {
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
                normalized_pix_costs[disp] = (cv_val - extremum_cost) / diff_cost;
            }

            // fill sampled ambiguity, compute integral
            amb_sum = 0;
            if (type_measure_min) {
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
            } else {
                for (int eta = 0; eta < nbr_etas; ++eta) {
                    for (int disp = 0; disp < n_disp; ++disp) {
                        amb_status = normalized_pix_costs[disp] >= (norm_extremum - r_etas(eta));
                        amb_sum += amb_status ? 1.f : 0.f;
                    }
                }
            }

            // fill integral ambiguity
            rw_amb(row, col) = amb_sum;

        }
    }

    delete[] normalized_pix_costs;

    return to_return;   

}
