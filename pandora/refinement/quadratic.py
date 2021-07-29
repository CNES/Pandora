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
This module contains functions associated to the quadratic method used in the refinement step.
"""

from typing import Dict, Tuple

import numpy as np
from json_checker import Checker, And
from numba import njit

import pandora.constants as cst
from . import refinement


@refinement.AbstractRefinement.register_subclass("quadratic")
class Quadratic(refinement.AbstractRefinement):
    """
    Quadratic class allows to perform the subpixel cost refinement step
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
        schema = {"refinement_method": And(str, lambda input: "quadratic")}

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the subpixel refinement method
        :return: None
        """
        print("Quadratic refinement method")

    @staticmethod
    @njit(cache=True)
    def refinement_method(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
        """
        Return the subpixel disparity and cost, by fitting a quadratic curve

        :param cost: cost of the values disp - 1, disp, disp + 1
        :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
        :param disp: the disparity
        :type disp: float
        :param measure: the type of measure used to create the cost volume
        :param measure: string = min | max
        :return: the disparity shift, the refined cost and the state of the pixel ( Information: \
        calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
        :rtype: float, float, int
        """

        if (np.isnan(cost[0])) or (np.isnan(cost[2])):
            # Bit 3 = 1: Information: calculations stopped at the pixel step, sub-pixel interpolation did not succeed
            return 0, cost[1], cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        inverse = 1
        if measure == "max":
            # Additive inverse : if a < b then -a > -b
            inverse = -1

        # Check if cost[disp] is the minimum cost (or maximum using similarity measure) before fitting
        # If not, interpolation is not applied
        if (inverse * cost[1] > inverse * cost[0]) or (inverse * cost[1] > inverse * cost[2]):
            return 0, cost[1], cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        # Solve the system: col = alpha * row ** 2 + beta * row + gamma
        alpha = (cost[0] - 2 * cost[1] + cost[2]) / 2
        beta = (cost[2] - cost[0]) / 2
        gamma = cost[1]

        # If the costs are close, the result of -b / 2a (minimum) is bounded between [-1, 1]
        # sub_disp is row
        sub_disp = min(1.0, max(-1.0, -beta / (2 * alpha)))

        # sub_cost is col
        sub_cost = (alpha * sub_disp ** 2) + (beta * sub_disp) + gamma

        return sub_disp, sub_cost, 0
