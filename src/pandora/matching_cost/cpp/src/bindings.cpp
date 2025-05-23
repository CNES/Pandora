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

#include <pybind11/pybind11.h>
#include "census.hpp"
#include "matching_cost.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matching_cost_cpp, m) {
    m.doc() = "Matching cost functions implemented in C++ with Pybind11";
    
    m.def(
        "compute_matching_costs", 
        &compute_matching_costs, 
        "Computes matching costs of images."
    );

    m.def(
        "reverse_cost_volume", 
        &reverse_cost_volume, 
        "Computes right cost volume from left cost volume."
    );

    m.def(
        "reverse_disp_range",
        &reverse_disp_range,
        "Computes the right disp range from the left one."
    );

}