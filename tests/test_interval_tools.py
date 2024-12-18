#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test all the methods in img_tools module.
"""

import numpy as np
import pytest

from pandora import interval_tools


class TestIntervalTools:
    """
    TestIntervalTools class allows to test all the methods in the module img_tools
    """

    @pytest.fixture()
    def border_left(self):
        return np.array([[0, 1], [0, 6], [1, 3], [2, 1], [3, 4], [3, 7], [4, 8], [5, 2], [5, 7], [6, 8]])

    @pytest.fixture()
    def border_right(self):
        return np.array([[0, 4], [0, 8], [1, 6], [2, 3], [3, 5], [3, 8], [4, 8], [5, 3], [5, 8], [6, 8]])

    @pytest.fixture()
    def gt_connection_matrix(self):
        """
        Ground truth for the connection matrix
        """
        gt_connection_matrix = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            ],
            dtype=bool,
        )
        return gt_connection_matrix

    @pytest.fixture()
    def gt_mask_modif(self):
        """
        Ground truth for the mask indicating which pixel have been regularized
        """
        gt_modif = np.array(
            [
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.bool_,
        )
        return gt_modif

    def test_create_connected_graph(self, border_left, border_right, gt_connection_matrix):
        """
        Test the method create_connected_graph for creating a connection matrix
        """

        graph = interval_tools.create_connected_graph(border_left, border_right, 2)

        # Checking if shape and data is the same
        np.testing.assert_array_equal(graph, gt_connection_matrix)

    def test_graph_regularization(self, border_left, border_right, gt_connection_matrix, gt_mask_modif):
        """
        Test the method for regularizing the graph with quantiles
        """
        # Creating test data for intervals and borders
        interval_inf = np.arange(1, 7 * 10 + 1, dtype=np.float32).reshape((7, 10))
        interval_sup = np.arange(5, 7 * 10 + 5, dtype=np.float32).reshape((7, 10))

        gt_left = np.array(
            [
                [1.0, 3.3, 3.3, 3.3, 3.3, 6.0, 3.3, 3.3, 3.3, 10.0],
                [11.0, 12.0, 13.0, 3.3, 3.3, 3.3, 3.3, 18.0, 19.0, 20.0],
                [21.0, 3.3, 3.3, 3.3, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.1, 35.1, 37.0, 38.4, 38.4, 40.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 38.5, 50.0],
                [51.0, 52.0, 53.1, 53.1, 55.0, 56.0, 57.0, 38.5, 38.5, 60.0],
                [61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 51.7, 70.0],
            ],
            dtype=np.float32,
        )

        gt_right = np.array(
            [
                [5.0, 26.7, 26.7, 26.7, 26.7, 10.0, 26.7, 26.7, 26.7, 14.0],
                [15.0, 16.0, 17.0, 26.7, 26.7, 26.7, 26.7, 22.0, 23.0, 24.0],
                [25.0, 26.7, 26.7, 26.7, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0],
                [35.0, 36.0, 37.0, 38.0, 39.9, 39.9, 41.0, 62.6, 62.6, 44.0],
                [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 68.0, 54.0],
                [55.0, 56.0, 57.9, 57.9, 59.0, 60.0, 61.0, 68.0, 68.0, 64.0],
                [65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 70.0, 74.0],
            ],
            dtype=np.float32,
        )

        # Regularizing the intervals
        reg_left, reg_right, graph = interval_tools.graph_regularization(
            interval_inf, interval_sup, border_left, border_right, gt_connection_matrix, 0.9
        )
        # Check if interval bounds and mask are equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(reg_left, gt_left, 1e-6, 1e-6)
        np.testing.assert_allclose(reg_right, gt_right, 1e-6, 1e-6)
        np.testing.assert_allclose(graph, gt_mask_modif, 1e-6, 1e-6)

    def test_interval_regularization(self, gt_mask_modif):
        """
        Test the method for interval_regularization
        """
        # Creating the test data
        interval_inf = np.arange(1, 7 * 10 + 1, dtype=np.float32).reshape((7, 10))
        interval_sup = np.arange(5, 7 * 10 + 5, dtype=np.float32).reshape((7, 10))

        ambiguity = np.array(
            [
                [1.0, 0.2, 0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 0.2],
                [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0],
                [1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                [1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ],
            dtype=np.float32,
        )

        # Ground truth interval bounds and mask
        gt_inf = np.array(
            [
                [1.0, 3.3, 3.3, 3.3, 3.3, 6.0, 3.3, 3.3, 3.3, 10.0],
                [11.0, 12.0, 13.0, 3.3, 3.3, 3.3, 3.3, 18.0, 19.0, 20.0],
                [21.0, 3.3, 3.3, 3.3, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.1, 35.1, 37.0, 38.4, 38.4, 40.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 38.5, 50.0],
                [51.0, 52.0, 53.1, 53.1, 55.0, 56.0, 57.0, 38.5, 38.5, 60.0],
                [61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 51.7, 70.0],
            ],
            dtype=np.float32,
        )

        gt_sup = np.array(
            [
                [5.0, 26.7, 26.7, 26.7, 26.7, 10.0, 26.7, 26.7, 26.7, 14.0],
                [15.0, 16.0, 17.0, 26.7, 26.7, 26.7, 26.7, 22.0, 23.0, 24.0],
                [25.0, 26.7, 26.7, 26.7, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0],
                [35.0, 36.0, 37.0, 38.0, 39.9, 39.9, 41.0, 62.6, 62.6, 44.0],
                [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 68.0, 54.0],
                [55.0, 56.0, 57.9, 57.9, 59.0, 60.0, 61.0, 68.0, 68.0, 64.0],
                [65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 70.0, 74.0],
            ],
            dtype=np.float32,
        )

        inf, sup, mask = interval_tools.interval_regularization(
            interval_inf,
            interval_sup,
            ambiguity,
            ambiguity_threshold=0.6,
            ambiguity_kernel_size=1,
            vertical_depth=2,
            quantile_regularization=0.9,
        )
        np.testing.assert_allclose(inf, gt_inf, 1e-6, 1e-6)
        np.testing.assert_allclose(sup, gt_sup, 1e-6, 1e-6)
        np.testing.assert_allclose(mask, gt_mask_modif, 1e-6, 1e-6)
