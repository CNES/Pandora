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
This module contains all the parameters related to the validity mask, defining each bit.
"""

# INVALID POINTS CONSTANTS
# INVALID
PANDORA_MSK_PIXEL_INVALID = 0b01111000011
# if bit 0 activated : The pixel is invalid : border of left image, OR nodata in left image
PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER = 1 << 0
# if bit 1 activated : The pixel is invalid : disparity range to explore is missing in right image, OR
#               nodata in right image
PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING = 1 << 1
# if bit 2 activated : Information : disparity range to explore is incomplete (borders reached in right image)
PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE = 1 << 2
# if bit 3 activated : Information : Computation stopped during pixelic step, under pixelic interpolation never ended
PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION = 1 << 3
# if bit 4 activated : Information : Occlusion was filled
PANDORA_MSK_PIXEL_FILLED_OCCLUSION = 1 << 4
# if bit 5 activated : Information : Mismatch was filled
PANDORA_MSK_PIXEL_FILLED_MISMATCH = 1 << 5
# if bit 6 activated : The pixel is invalid : invalidated by validity mask of left image
PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT = 1 << 6
# if bit 7 activated : The pixel is invalid : invalidated by validity mask of right image
PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT = 1 << 7
# if bit 8 activated : The pixel is invalid : pixel in occlusion area
PANDORA_MSK_PIXEL_OCCLUSION = 1 << 8
# if bit 9 activate : The pixel is invalid : mismatch
PANDORA_MSK_PIXEL_MISMATCH = 1 << 9
# if bit 10 activate : Information : Nodata was filled
PANDORA_MSK_PIXEL_FILLED_NODATA = 1 << 10
