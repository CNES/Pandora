# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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

# pylint: skip-file

def compute_ambiguity_and_sampled_ambiguity(cv, etas, nbr_etas, grids, disparity_range, sample_ambiguity):
    """
    Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param etas: range between eta_min and eta_max with step eta_step
    :type etas: np.ndarray
    :param nbr_etas: nuber of etas
    :type nbr_etas: int
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray
    :param sample_ambiguity: whether to return the sampled ambiguity along with the ambiguity
    :type sample_ambiguity: bool
    :return: the normalized ambiguity and sampled ambiguity
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
    """
    return None, None

def compute_interval_bounds(cv, disp_interval, possibility_threshold, type_factor, grids, disparity_range):
    """
    Computes interval bounds on the disparity.

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param disp_interval: disparity data
    :type disp_interval: 1D np.ndarray (disp,)
    :param possibility_threshold: possibility threshold used for interval computation
    :type possibility_threshold: float
    :param type_factor: Either 1 or -1. Used to adapt the possibility computation to max or min measures
    :type type_factor: float
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray

    :return: the infimum and supremum (not regularized) of the set containing the true disparity
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32)
    """
    return None, None

def compute_risk_and_sampled_risk(cv, sampled_ambiguity, etas, nbr_etas, grids, disparity_range, sample_risk):
    """
    Computes minimum and maximum risk, and sampled_risk if asked to.

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param sampled_ambiguity: sampled cost volume ambiguity
    :type sampled_ambiguity: 3D np.ndarray (row, col, eta)
    :param etas: range between eta_min and eta_max with step eta_step
    :type etas: np.ndarray
    :param nbr_etas: nuber of etas
    :type nbr_etas: int
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray
    :param sample_risk: whether or not to compute and return the sampled risk
    :type sample_risk: bool
    :return: the risk, the disp min and max and sampled risk if asked
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32, \
    2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32, \
    3D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
    """
    return None, None, None, None
