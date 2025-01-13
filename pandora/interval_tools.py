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

from typing import Tuple

import numpy as np

from .cpp import interval_tools_cpp

create_connected_graph = interval_tools_cpp.create_connected_graph

graph_regularization = interval_tools_cpp.graph_regularization


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
