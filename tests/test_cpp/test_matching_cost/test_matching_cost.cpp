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
This module contains tests associated to matching cost computation.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NAN std::nanf("")

#include <doctest.h>
#include <limits>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;
#include "matching_cost.hpp"


TEST_CASE("Cost volume inversion test 1"){

    py::scoped_interpreter guard{};

    size_t n_rows = 1;
    size_t n_cols = 6;
    size_t n_disp = 4;

    py::array_t<float> cv = py::array_t<float>({n_rows, n_cols, n_disp});
    auto rw_cv = cv.mutable_unchecked<3>();
    
    // Assuming the 6x1 images : 
    // left:  A B C D E F
    // right: G H I J K L

    // With disp_min = 1, disp_max = 4
    // CV left contains (row col disp):
    // { 
    //   { // ROW 1
    //     {AH, AI, AJ, AK} // COL 1
    //     {BI, BJ, BK, BL} // COL 2
    //     {CJ, CK, CL, **} // COL 3
    //     {DK, DL, **, **} // COL 4
    //     {EL, **, **, **} // COL 5
    //     {**, **, **, **} // COL 6
    //   }
    // }

    // With disp_min = -4, disp_max = -1
    // CV right then contains :
    // { 
    //   { // ROW 1
    //     {**, **, **, **} // COL 1
    //     {**, **, **, AH} // COL 2
    //     {**, **, AI, BI} // COL 3
    //     {**, AJ, BJ, CJ} // COL 4
    //     {AK, BK, CK, DK} // COL 5
    //     {BL, CL, DL, EL} // COL 6
    //   }
    // }

    float left_cv_arr[1][6][4] = {
        {
            { 12,  13,  14,  15},
            { 23,  24,  25,  26},
            { 34,  35,  36, NAN},
            { 45,  46, NAN, NAN},
            { 56, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN}
        }
    };

    float ref_right_cv_arr[1][6][4] = {
        {
            { NAN, NAN, NAN, NAN},
            { NAN, NAN, NAN,  12},
            { NAN, NAN,  13,  23},
            { NAN,  14,  24,  34},
            {  15,  25,  35,  45},
            {  26,  36,  46,  56}
        }
    };

    for (int i = 0; i < 1; ++i)
        for (int j = 0; j < 6; ++j)
            for (int k = 0; k < 4; ++k)
                rw_cv(i,j,k) = left_cv_arr[i][j][k];

    // float min_disp = 1; // unused
    float max_disp = 4;

    // right_min_disp = -left_max_disp
    py::array_t<float> right_cv = reverse_cost_volume(cv, -max_disp);

    auto r_right_cv = right_cv.unchecked<3>();

    // Uncomment to show outputs in console (debug)

    //for (int i = 0; i < n_rows; ++i) {
    //    for (int j = 0; j < n_cols; ++j) {
    //        for (int k = 0; k < n_disp; ++k) {
    //            std::cout << r_right_cv(i,j,k) << ", ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << std::endl;
    //}

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            for (size_t k = 0; k < n_disp; ++k)
                if (std::isnan(r_right_cv(i, j, k)))
                    // check that ref is also a nan
                    CHECK( std::isnan(ref_right_cv_arr[i][j][k]) );
                else
                    // check that both values are equal
                    CHECK(r_right_cv(i, j, k) == ref_right_cv_arr[i][j][k]);

}


TEST_CASE("Cost volume inversion test 2"){

    py::scoped_interpreter guard{};

    size_t n_rows = 1;
    size_t n_cols = 6;
    size_t n_disp = 5;

    py::array_t<float> cv = py::array_t<float>({n_rows, n_cols, n_disp});
    auto rw_cv = cv.mutable_unchecked<3>();
    
    // Assuming the 6x1 images : 
    // left:  A B C D E F
    // right: G H I J K L

    // With disp_min = -2, disp_max = 2
    // CV left contains (row col disp):
    // { 
    //   { // ROW 1
    //     {**, **, AG, AH, AI} // COL 1
    //     {**, BG, BH, BI, BJ} // COL 2
    //     {CG, CH, CI, CJ, CK} // COL 3
    //     {DH, DI, DJ, DK, DL} // COL 4
    //     {EI, EJ, EK, EL, **} // COL 5
    //     {FJ, FK, FL, **, **} // COL 6
    //   }
    // }

    // With disp_min = -2, disp_max = 2
    // CV right then contains :
    // { 
    //   { // ROW 1
    //     {**, **, AG, BG, CG} // COL 1
    //     {**, AH, BH, CH, DH} // COL 2
    //     {AI, BI, CI, DI, EI} // COL 3
    //     {BJ, CJ, DJ, EJ, FJ} // COL 4
    //     {CK, DK, EK, FK, **} // COL 5
    //     {DL, EL, FL, **, **} // COL 6
    //   }
    // }

    float left_cv_arr[1][6][5] = { 
        {
            {NAN, NAN,  11,  12,  13},
            {NAN,  21,  22,  23,  24},
            { 31,  32,  33,  34,  35},
            { 42,  43,  44,  45,  46},
            { 53,  54,  55,  56, NAN},
            { 64,  65,  66, NAN, NAN}
        }
    };

    float ref_right_cv_arr[1][6][5] = { 
        {
            {NAN, NAN,  11,  21,  31},
            {NAN,  12,  22,  32,  42},
            { 13,  23,  33,  43,  53},
            { 24,  34,  44,  54,  64},
            { 35,  45,  55,  65, NAN},
            { 46,  56,  66, NAN, NAN}
        }
    };

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            for (size_t k = 0; k < n_disp; ++k)
                rw_cv(i,j,k) = left_cv_arr[i][j][k];

    // float min_disp = -2; // unused
    float max_disp = 2;

    // right_min_disp = -left_max_disp
    py::array_t<float> right_cv = reverse_cost_volume(cv, -max_disp);

    auto r_right_cv = right_cv.unchecked<3>();

    // Uncomment to show outputs in console (debug)

    //for (int i = 0; i < n_rows; ++i) {
    //    for (int j = 0; j < n_cols; ++j) {
    //        for (int k = 0; k < n_disp; ++k) {
    //            std::cout << r_right_cv(i,j,k) << ", ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << std::endl;
    //}

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            for (size_t k = 0; k < n_disp; ++k)
                if (std::isnan(r_right_cv(i, j, k)))
                    // check that ref is also a nan
                    CHECK( std::isnan(ref_right_cv_arr[i][j][k]) );
                else
                    // check that both values are equal
                    CHECK(r_right_cv(i, j, k) == ref_right_cv_arr[i][j][k]);

}

