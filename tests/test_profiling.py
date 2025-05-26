# type:ignore
#!/usr/bin/env python
# coding: utf8
#
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
#
"""
This module contains functions to test the disparity map validation step.
"""

import unittest
import pytest

from json_checker.core.exceptions import DictCheckerError, MissKeyCheckerError

from pandora.profiler import Profiler


class TestProfiling(unittest.TestCase):
    """
    TestValidation class allows to test all the methods in the module Validation
    """

    def test_profiling_no_parameters_needed(self):
        """
        Test that there's no crash when not giving the profiling key
        """
        Profiler.enable_from_config({"inputs": {}, "pipeline": {}})  # type: ignore

    def test_profiling_correct_inputs(self):
        """
        Test that there's no crash when giving profiling as a bool or dict with correct inputs
        """
        Profiler.enable_from_config({"profiling": True})  # type: ignore
        Profiler.enable_from_config({"profiling": False})  # type: ignore
        Profiler.enable_from_config({"profiling": {}})  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_graphs": True,
                }
            }
        )  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_raw_data": True,
                }
            }
        )  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_graphs": False,
                    "save_raw_data": True,
                }
            }
        )  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_graphs": True,
                    "save_raw_data": False,
                }
            }
        )  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_graphs": False,
                    "save_raw_data": False,
                }
            }
        )  # type: ignore
        Profiler.enable_from_config(
            {
                "profiling": {
                    "save_graphs": True,
                    "save_raw_data": True,
                }
            }
        )  # type: ignore

    def test_profiling_anything_else_crashes(self):
        """
        Test that there's a crash with invalid values
        """
        with pytest.raises(MissKeyCheckerError):
            Profiler.enable_from_config(
                {
                    "profiling": {
                        "display_graphs": True,
                    }
                }
            )  # type: ignore
        with pytest.raises(MissKeyCheckerError):
            Profiler.enable_from_config(
                {
                    "profiling": {
                        "wrong key": True,
                    }
                }
            )  # type: ignore
        for value in [134, "Something's wrong", [True, False], None]:
            with pytest.raises(TypeError):
                Profiler.enable_from_config({"profiling": value})  # type: ignore
            with pytest.raises(DictCheckerError):
                Profiler.enable_from_config(
                    {
                        "profiling": {
                            "save_graphs": value,
                        }
                    }
                )  # type: ignore
            with pytest.raises(DictCheckerError):
                Profiler.enable_from_config(
                    {
                        "profiling": {
                            "save_raw_data": value,
                        }
                    }
                )  # type: ignore
