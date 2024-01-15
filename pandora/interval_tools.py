#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions associated to confidence intervals.
"""

import os
from ast import literal_eval
from typing import Tuple

import numpy as np
from numba import njit, prange


@njit("b1[:,:](i8[:,:], i8[:,:], i8)", parallel=literal_eval(os.environ.get("PANDORA_NUMBA_PARALLEL", "True")))
def create_connected_graph(border_left: np.ndarray, border_right: np.ndarray, depth: int) -> np.ndarray:
    """
    Create a boolean connection matrix from segment coordinates

    :param border_left: array containing the coordinates of segments left border
    :type border_left: (n, 2) np.ndarray where n is the number of segments
    :param border_right: array containing the coordinates of segments right border
    :type border_right: (n, 2) np.ndarray where n is the number of segments
    :param depth: the depth for regularization. It corresponds to the number of rows
        to explore below and above.
    :return: A symmetric boolean matrix of shape (n, n). 1 indicating that the segment are connected
    :rtype: np.ndarray of shape (n, n)
    """
    # border_left and border_right are already sorted by argwhere
    # we only need to create a connection graph by looking at neighboors from below
    n_segments = len(border_left)

    if depth == 0:
        aggregated_graph = np.eye(n_segments, dtype=np.bool_)
    else:
        connection_graph = np.full((n_segments, n_segments), False, dtype=np.bool_)
        for i in prange(n_segments):  # pylint: disable=not-an-iterable
            row_i = border_left[i, 0]
            for k in range(i + 1, n_segments):
                if border_left[k, 0] == row_i:
                    continue
                if border_left[k, 0] > row_i + 1:
                    break
                if (border_left[k, 1] <= border_right[i, 1]) & (border_right[k, 1] >= border_left[i, 1]):
                    connection_graph[i, k] = connection_graph[k, i] = True

        aggregated_graph = np.full((n_segments, n_segments), False, dtype=np.bool_)
        for i in prange(connection_graph.shape[0]):  # pylint: disable=not-an-iterable
            list_lines = connection_graph[i, :].copy()
            for _ in range(1, depth):
                new_points = connection_graph[list_lines, :].copy()
                for j in prange(connection_graph.shape[0]):  # pylint: disable=not-an-iterable
                    list_lines[j] = np.bitwise_or(new_points[:, j].any(), list_lines[j])
            aggregated_graph[i, :] = list_lines.copy()
            aggregated_graph[i, i] = 1
    return aggregated_graph


@njit(
    "Tuple([f4[:,:],f4[:,:],b1[:,:]])(f4[:,:],f4[:,:],i8[:,:],i8[:,:],b1[:,:],f8)",
    parallel=literal_eval(os.environ.get("PANDORA_NUMBA_PARALLEL", "True")),
)
def graph_regularization(
    interval_inf: np.ndarray,
    interval_sup: np.ndarray,
    border_left: np.ndarray,
    border_right: np.ndarray,
    connection_graph: np.ndarray,
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regularize the intervals based on quantiles and a given connection graph.

    :param interval_inf: The lower estimation of the disparity to regularize
    :type interval_inf: (row, col) np.ndarray
    :param interval_sup: The upper estimation of the disparity to regularize
    :type interval_sup: (row, col) np.ndarray
    :param border_left: array containing the coordinates of segments left border
    :type border_left: (n, 2) np.ndarray where n is the number of segments
    :param border_right: array containing the coordinates of segments right border
    :type border_right: (n, 2) np.ndarray where n is the number of segments
    :param connection graph: A matrix indicating if the segments (n in total) are connected
    :type connection graph: (n, n) boolean symmetric np.ndarray
    :param quantile: Which quantile to select for the regularized value
    :type quantile: float. 0 <= quantile <= 1
    :return: The regularized inf and sup of the disparity, and a boolean mask indicating regularization
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    interval_inf_reg = interval_inf.copy()
    interval_sup_reg = interval_sup.copy()
    mask_regularization = np.full(interval_inf.shape, False, dtype=np.bool_)

    for i in prange(connection_graph.shape[0]):  # pylint: disable=not-an-iterable
        left_i, right_i = border_left[connection_graph[i, :]], border_right[connection_graph[i, :]]

        # Contains the lengths of the segments
        n_pixels = np.hstack((np.array([0]), (right_i[:, 1] - left_i[:, 1] + 1).cumsum()))
        agg_inf = np.full(n_pixels[-1], 0, dtype=np.float32)
        agg_sup = np.full(n_pixels[-1], 0, dtype=np.float32)

        for j in range(len(n_pixels) - 1):
            agg_inf[n_pixels[j] : n_pixels[j + 1]] = interval_inf[left_i[j, 0], left_i[j, 1] : right_i[j, 1] + 1]
            agg_sup[n_pixels[j] : n_pixels[j + 1]] = interval_sup[left_i[j, 0], left_i[j, 1] : right_i[j, 1] + 1]

        interval_inf_reg[border_left[i, 0], border_left[i, 1] : border_right[i, 1] + 1] = np.nanquantile(
            agg_inf, 1 - quantile
        )
        interval_sup_reg[border_left[i, 0], border_left[i, 1] : border_right[i, 1] + 1] = np.nanquantile(
            agg_sup, quantile
        )
        mask_regularization[border_left[i, 0], border_left[i, 1] : border_right[i, 1] + 1] = True

    return interval_inf_reg, interval_sup_reg, mask_regularization


def interval_regularization(
    interval_inf: np.ndarray,
    interval_sup: np.ndarray,
    ambiguity: np.ndarray,
    ambiguity_threshold: float,
    ambiguity_kernel_size: int,
    vertical_depth: int = 0,
    quantile_regularization: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regularize interval bounds in ambiguous zones.

    :param interval_inf: lower bound of the confidence interval
    :type cv: 2D np.ndarray (row, col)
    :param interval_sup: upper bound of the confidence interval
    :type cv: 2D np.ndarray (row, col)
    :param ambiguity: ambiguity confidence map
    :type cv: 2D np.ndarray (row, col)
    :param ambiguity_threshold: threshold used for detecting ambiguous zones
    :type ambiguity_threshold: float
    :param ambiguity_kernel_size: number of columns for the minimitive kernel applied to ambiguity
    :type ambiguity_kernel_size: int
    :param vertical_depth: The number of lines above and below to look for adjacent segment during the regularization
    :type vertical_depth: int >= 0
    :param quantile_regularization: The quantile used for selecting the disparity value in the regularization step
    :type quantile_regularization: float between 0 and 1

    :return: the regularized infimum and supremum of the set containing the true disparity
        and the mask of pixel that have been regularized
    :rtype: Tuple(2D np.array (row, col) dtype = float32,
                  2D np.array (row, col) dtype = float32,
                  2D np.array (row, col) dtype = np.bool)
    """

    n_row, _ = ambiguity.shape
    pad = ambiguity_kernel_size // 2
    minimized_conf_from_amb = np.hstack((np.ones((n_row, pad)), ambiguity, np.ones((n_row, pad))))
    # H-stacking is to conserve the same shape after view.
    # Because we take the minimum afterwards, it needs to be ones
    minimized_conf_from_amb = np.nanmin(
        np.lib.stride_tricks.sliding_window_view(minimized_conf_from_amb, ambiguity_kernel_size, axis=1),
        axis=-1,
    )

    # Final column to 1 and H-stacking is to unsure that
    # we always start with a left border and end with a right border
    minimized_conf_from_amb[:, -1] = 1
    border = np.diff(
        np.hstack([np.ones((minimized_conf_from_amb.shape[0], 1)), minimized_conf_from_amb >= ambiguity_threshold]),
        axis=-1,
    )
    border_left = np.argwhere(border == -1)
    border_right = np.argwhere(border == 1)

    # If border[i,j]==1, then minimized_conf_from_amb[i,j]>=ambiguity_threshold and we keep i, j-1
    border_right[:, 1] = border_right[:, 1] - 1

    graph = create_connected_graph(border_left, border_right, vertical_depth)

    return graph_regularization(interval_inf, interval_sup, border_left, border_right, graph, quantile_regularization)
