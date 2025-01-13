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
This module contains functions associated to the quadratic method used in the refinement step.
"""

from typing import Dict, Tuple
import numpy as np

from json_checker import And, Checker

import pandora.constants as cst

from .cpp import refinement_cpp
from . import refinement


@refinement.AbstractRefinement.register_subclass("quadratic")
class Quadratic(refinement.AbstractRefinement):
    """
    Quadratic class allows to perform the subpixel cost refinement step
    """

    @staticmethod
    def refinement_method(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
        return refinement_cpp.quadratic_refinement_method(
            cost, disp, measure, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION
        )

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
