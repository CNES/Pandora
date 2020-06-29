#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
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

import numpy as np
import xarray as xr
from typing import Dict


def get_margins(disp_min: int, disp_max: int, cfg: Dict[str, dict]) -> xr.DataArray:
    """
    Calculates the margins for the reference and secondary images according to the configuration

    :param disp_min: minimal disparity
    :type disp_min: int
    :param disp_max: maximal disparity
    :type disp_max: int
    :param cfg: user configuration
    :type cfg: dict of dict
    :return: margin for the images
    :rtype: 2D (image, corner) DataArray, with the dimensions image = ['ref_margin', 'sec_margin'],
        corner = ['left', 'up', 'right', 'down']
    """
    margin = xr.DataArray(np.zeros((2, 4), dtype=int), coords=[['ref_margin', 'sec_margin'], ['left','up','right','down']],
                          dims=['image', 'corner'])
    margin.name = 'Margins'

    # Margins for the reference image and for the secondary image

    # Pandora margins depends on the steps configured
    if cfg['optimization']['optimization_method'] == 'sgm':
        # SGM margin includes the census, vfit and median filter margins
        sgm_margins = 40
        r_marg = [sgm_margins + disp_max, sgm_margins, sgm_margins - disp_min, sgm_margins]
        s_marg = [sgm_margins - disp_min, sgm_margins, sgm_margins + disp_max, sgm_margins]

    else:
        r_marg = np.array([disp_max, 0, -disp_min, 0])
        s_marg = np.array([-disp_min, 0, + disp_max, 0])

        if cfg['stereo']['window_size'] != 1:
            r_marg += int(cfg['stereo']['window_size'] / 2)
            s_marg += int(cfg['stereo']['window_size'] / 2)

        if cfg['refinement']['refinement_method'] == 'vfit':
            r_marg[0] += 1
            r_marg[2] += 1
            s_marg[0] += 1
            s_marg[2] += 1

        if cfg['filter']['filter_method'] == 'median':
            r_marg += int(cfg['filter']['filter_size'] / 2)
            s_marg += int(cfg['filter']['filter_size'] / 2)

    # Same margin for ref and sec: take the larger
    same_margin = list(map(lambda x: max(x[0], x[1]), zip(r_marg, s_marg)))
    margin.loc[dict(image='ref_margin')] = same_margin
    margin.loc[dict(image='sec_margin')] = same_margin

    # Save disp_min and disp_max
    margin.attrs['disp_min'] = disp_min
    margin.attrs['disp_max'] = disp_max

    return margin
