/* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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

/*
This module contains tests associated to confidence.
*/


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;
#include "cost_volume_confidence_tools.hpp"


TEST_CASE("test min max function"){

    py::scoped_interpreter guard{};

    size_t n_rows = 1;
    size_t n_cols = 3;
    size_t n_disp = 4;

    py::array_t<float> cv_ = py::array_t<float>({n_rows, n_cols, n_disp});
    auto unchecked_cv = cv_.mutable_unchecked<3>();

    unchecked_cv(0, 0, 0) = 39; unchecked_cv(0, 0, 1) = 28.03f; unchecked_cv(0,0,2) = 28; unchecked_cv(0,0,3) = 34.5f;
    unchecked_cv(0,1,0) = 49; unchecked_cv(0,1,1) = 34; unchecked_cv(0,1,2) = 41.5f; unchecked_cv(0,1,3) = 34.1f;
    unchecked_cv(0,2,0) = std::nanf(""); unchecked_cv(0,2,1) = std::nanf(""); unchecked_cv(0,2,2) = std::nanf(""); unchecked_cv(0,2,3) = std::nanf("");


    float min_cost_ref;
    float max_cost_ref;
    min_cost_ref = 28;
    max_cost_ref = 49;

    std::vector<float> rw_min_img_ref = {28, 34, std::nanf("")};
    std::vector<float> rw_max_img_ref = {39, 49, std::nanf("")};

    auto [min_cost, max_cost, rw_min_img, rw_max_img] = min_max_cost(unchecked_cv, n_rows, n_cols, n_disp);

    CHECK(min_cost == doctest::Approx(28));
    CHECK(max_cost == doctest::Approx(49));

    CHECK(rw_min_img(0, 0) == doctest::Approx(rw_min_img_ref[0]));
    CHECK(rw_min_img(0, 1) == doctest::Approx(rw_min_img_ref[1]));
    CHECK(std::isnan(rw_min_img(0, 2)));

    CHECK(rw_max_img(0, 0) == doctest::Approx(rw_max_img_ref[0]));
    CHECK(rw_max_img(0, 1) == doctest::Approx(rw_max_img_ref[1]));
    CHECK(std::isnan(rw_max_img(0, 2)));

    CHECK(min_cost == doctest::Approx(min_cost_ref));
    CHECK(max_cost == doctest::Approx(max_cost_ref));

}

