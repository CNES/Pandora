#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
This module contains the required libraries and softwares allowing to execute the software,
and setup elements to configure and identify the software.
setup.py kept for different version compatibility
All packaging in setup.cfg
"""

from setuptools import setup

try:
    setup()
except Exception:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of pip, setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U pip setuptools setuptools_scm wheel\n\n"
    )
    raise
