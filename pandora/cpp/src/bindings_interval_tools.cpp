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
#include "interval_tools.hpp"

namespace py = pybind11;

PYBIND11_MODULE(interval_tools_cpp, m) {
    m.doc() = "Interval tools functions implemented in C++ with Pybind11";

    m.def(
        "create_connected_graph", 
        &create_connected_graph, 
        ""
    );
    m.def(
        "graph_regularization", 
        &graph_regularization, 
        ""
    );

}
