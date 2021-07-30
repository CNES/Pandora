#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the vfit method used in the refinement step.
"""

from typing import Dict, Tuple

import numpy as np
from json_checker import Checker, And
from numba import njit

import pandora.constants as cst
from . import refinement


@refinement.AbstractRefinement.register_subclass("vfit")
class Vfit(refinement.AbstractRefinement):
    """
    Vfit class allows to perform the subpixel cost refinement step
    """

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._refinement_method_name = str(self.cfg["refinement_method"])

    @staticmethod
    def check_conf(**cfg: str) -> Dict[str, str]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: refinement configuration
        :type cfg: dict
        :return cfg: refinement configuration updated
        :rtype: dict
        """
        schema = {"refinement_method": And(str, lambda x: "vfit")}

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the subpixel refinement method
        :return: None
        """
        print("Vfit refinement method")

    @staticmethod
    @njit()
    def refinement_method(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
        """
        Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)

        :param cost: cost of the values disp - 1, disp, disp + 1
        :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
        :param disp: the disparity
        :type disp: float
        :param measure: the type of measure used to create the cost volume
        :param measure: string = min | max
        :return: the disparity shift, the refined cost and the state of the pixel( Information: \
        calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
        :rtype: float, float, int
        """
        if (np.isnan(cost[0])) or (np.isnan(cost[2])):
            # Information: calculations stopped at the pixel step, sub-pixel interpolation did not succeed
            return 0, cost[1], cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        inverse = 1
        if measure == "max":
            # Additive inverse : if a < b then -a > -b
            inverse = -1

        # Check if cost[disp] is the minimum cost (or maximum using similarity measure) before matching a symmetric V
        # shape, if not, interpolation is not applied
        if (inverse * cost[1] > inverse * cost[0]) or (inverse * cost[1] > inverse * cost[2]):
            return 0, cost[1], cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        # The problem is to approximate sub_cost function with an affine function: y = a * x + origin
        # Calculate the slope
        a = cost[2] - cost[1]

        # Compare the difference disparity between (cost[0]-cost[1]) and (cost[2]-cost[1]): the highest cost is used
        if (inverse * cost[0]) > (inverse * cost[2]):
            a = cost[0] - cost[1]

        if abs(a) < 1.0e-15:
            return 0, cost[1], 0

        # Problem is resolved with tangents equality, due to the symmetric V shape of 3 points (cv0, cv2 and (x,y))
        # sub_disp is dx
        sub_disp = (cost[0] - cost[2]) / (2 * a)

        # sub_cost is y
        sub_cost = a * (sub_disp - 1) + cost[2]

        return sub_disp, sub_cost, 0
