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
This module contains the function which defines the images margins.
"""

from typing import Dict

import numpy as np
import xarray as xr


def get_margins(disp_min: int, disp_max: int, cfg: Dict[str, dict]) -> xr.Dataset:
    """
    Calculates the margins for the left and right images according to the
    steps present on the pipeline.

    - The margins are initialized at [+-disp_max, 0, -+disp_min, 0] for
      right and left images.
    - For matching_cost computation, a margin of half the window size is considered.
    - For vfit refinement, a margin of 1 pixel size is considered at positions 0 and 2.
    - For median filter, a margin of half the filter size is considered.
    - For SGM or 3SGM optimization, since infinite margins are theoretically
      required, a relatively high value (40 pixels) is considered.
    - The maximum margin obtained between (matching_cost + vfit
      refinement + median filter) and SGM/3SGM will be considered.

    :param disp_min: minimal disparity
    :type disp_min: int
    :param disp_max: maximal disparity
    :type disp_max: int
    :param cfg: user configuration
    :type cfg: dict of dict
    :return: margin for the images, 2D (image, corner) DataArray, with the dimensions image = \
     ['left_margin', 'right_margin'], corner = ['left', 'up', 'right', 'down']
    :rtype: xr.dataset
    """
    corner = ["left", "up", "right", "down"]
    data = np.zeros(len(corner))
    col = np.arange(len(corner))
    margin = xr.Dataset({"left_margin": (["col"], data)}, coords={"col": col})
    margin["right_margin"] = xr.DataArray(data, dims=["col"])

    # Margins for the left image and for the right image

    r_marg = np.array([disp_max, 0, -disp_min, 0])
    s_marg = np.array([-disp_min, 0, +disp_max, 0])

    if cfg["matching_cost"]["window_size"] != 1:
        r_marg += int(cfg["matching_cost"]["window_size"] / 2)
        s_marg += int(cfg["matching_cost"]["window_size"] / 2)

    if "refinement" in cfg:
        if cfg["refinement"]["refinement_method"] == "vfit":
            r_marg[0] += 1
            r_marg[2] += 1
            s_marg[0] += 1
            s_marg[2] += 1

    if "filter" in cfg:
        if cfg["filter"]["filter_method"] == "median":
            r_marg += int(cfg["filter"]["filter_size"] / 2)
            s_marg += int(cfg["filter"]["filter_size"] / 2)

    # Pandora margins depends on the steps configured
    if "optimization" in cfg:
        if cfg["optimization"]["optimization_method"] in ["sgm", "3sgm"]:
            # Since SGM theoretically requires infinite margins, they are fixed to a
            # relatively high value
            sgm_margins = 40
            r_marg_optim = [
                sgm_margins + disp_max,
                sgm_margins,
                sgm_margins - disp_min,
                sgm_margins,
            ]
            s_marg_optim = [
                sgm_margins - disp_min,
                sgm_margins,
                sgm_margins + disp_max,
                sgm_margins,
            ]
            # Select the maximum margin between optimization and the rest
            # of Pandora's pipeline steps
            for idx, _ in enumerate(s_marg):
                r_marg[idx] = np.max([r_marg[idx], r_marg_optim[idx]])
                s_marg[idx] = np.max([s_marg[idx], s_marg_optim[idx]])

    # Same margin for left and right: take the larger
    same_margin = list(map(lambda input: max(input[0], input[1]), zip(r_marg, s_marg)))
    margin["left_margin"].data = same_margin
    margin["right_margin"].data = same_margin

    # Save disp_min and disp_max
    margin.attrs["disp_min"] = disp_min
    margin.attrs["disp_max"] = disp_max

    return margin
