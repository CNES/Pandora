#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains all the parameters related to the validity mask, defining each bit.
"""

from enum import IntFlag, auto

from numpy.typing import NDArray


class Criteria(IntFlag):
    """
    Validity mask criteria (bitfield per pixel).
    """

    PANDORA_VALID = 0

    # if bit 0 activated : The pixel is invalid : border of left image, OR nodata in left image
    PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER = auto()
    # if bit 1 activated : The pixel is invalid : disparity range to explore is missing in right image, OR
    #               nodata in right image
    PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING = auto()
    # if bit 2 activated : Information : disparity range to explore is incomplete (borders reached in right image)
    PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE = auto()
    # if bit 3 activated : Information : Computation stopped during pixelic step, under pixelic interpolation never ended
    PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION = auto()
    # if bit 4 activated : Information : Occlusion was filled
    PANDORA_MSK_PIXEL_FILLED_OCCLUSION = auto()
    # if bit 5 activated : Information : Mismatch was filled
    PANDORA_MSK_PIXEL_FILLED_MISMATCH = auto()
    # if bit 6 activated : The pixel is invalid : invalidated by validity mask of left image
    PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT = auto()
    # if bit 7 activated : The pixel is invalid : invalidated by validity mask of right image
    PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT = auto()
    # if bit 8 activated : The pixel is invalid : pixel in occlusion area
    PANDORA_MSK_PIXEL_OCCLUSION = auto()
    # if bit 9 activate : The pixel is invalid : mismatch
    PANDORA_MSK_PIXEL_MISMATCH = auto()
    # if bit 10 activate : Information : Nodata was filled
    PANDORA_MSK_PIXEL_FILLED_NODATA = auto()
    # if bit 11 activate : Information : Interval was in a regularization zone
    PANDORA_MSK_PIXEL_INTERVAL_REGULARIZED = auto()
    # if bit 12 activate : Information : Interval in the image touches a border or contains 1 or more nodata
    PANDORA_MSK_PIXEL_INCOMPLETE_VARIABLE_DISPARITY_RANGE = auto()

    # INVALID (combination of invalid bits: 0, 1, 6, 7, 8, 9 — same as former 0b01111000011)
    PANDORA_MSK_PIXEL_INVALID = (
        PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
        | PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
        | PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
        | PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
        | PANDORA_MSK_PIXEL_OCCLUSION
        | PANDORA_MSK_PIXEL_MISMATCH
    )

    def is_in(self, array: NDArray):
        """Returns a bool array, where True if Criteria value is part of array element."""
        return array & self._value_ == self._value_
