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
This module defines the output tree structure.
"""

import os

OTD = {
    "left_disparity.tif": ".",
    "right_disparity.tif": ".",
    "left_confidence_measure.tif": ".",
    "right_confidence_measure.tif": ".",
    "left_validity_mask.tif": ".",
    "right_validity_mask.tif": ".",
    # Configuration
    "config.json": "./cfg",
    "command_line.txt": "./cfg",
}


def get_out_dir(key: str) -> str:
    """
    Return the output directory
    :param key: output product
    """
    return OTD[key]


def get_out_file_path(key: str) -> str:
    """
    Return the output file path
    :param key: output product
    """
    return os.path.join(get_out_dir(key), key)
