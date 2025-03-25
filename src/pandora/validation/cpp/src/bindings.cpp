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
#include "interpolated_disparity.hpp"

namespace py = pybind11;

PYBIND11_MODULE(validation_cpp, m) {
    m.doc() = "Validation functions implemented in C++ with Pybind11";
    
    m.def(
        "interpolate_occlusion_sgm", 
        &interpolate_occlusion_sgm, 
        ""
    );
    m.def(
        "interpolate_mismatch_sgm", 
        &interpolate_mismatch_sgm, 
        ""
    );
    
    m.def(
        "interpolate_occlusion_mc_cnn", 
        &interpolate_occlusion_mc_cnn, 
        ""
    );
    m.def(
        "interpolate_mismatch_mc_cnn", 
        &interpolate_mismatch_mc_cnn, 
        ""
    );

}