# type:ignore
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
This module contains functions to test the disparity denoiser filter.
"""

import pytest
from json_checker import MissKeyCheckerError, DictCheckerError

import pandora.filter as flt


class TestDisparityDenoiser:
    """Test DisparityDenoiser"""

    def disparity_denoiser_filter(self):
        """
        Instantiate a disparity_denoiser Filter.
        """
        return flt.AbstractFilter(cfg={"filter_method": "disparity_denoiser"})

    def test_check_conf_with_default_values(self):
        """Test check conf with default values"""
        filter_config = {
            "filter_method": "disparity_denoiser",
            "filter_size": 11,
            "sigma_euclidian": 4.0,
            "sigma_color": 100.0,
            "sigma_planar": 12.0,
            "sigma_grad": 1.5,
        }

        disparity_denoiser = flt.AbstractFilter(cfg=filter_config)

        assert disparity_denoiser.cfg == filter_config

    def test_check_conf_with_new_values(self):
        """Test check conf with new values"""
        filter_config = {
            "filter_method": "disparity_denoiser",
            "filter_size": 10,
            "sigma_euclidian": 5.0,
            "sigma_color": 90.0,
            "sigma_planar": 10.0,
            "sigma_grad": 1.0,
        }

        disparity_denoiser = flt.AbstractFilter(cfg=filter_config)

        assert disparity_denoiser.cfg == filter_config

    @pytest.mark.parametrize("missing_key", ["filter_method"])
    def test_check_conf_fails_when_is_missing_mandatory_key(self, missing_key):
        """When a mandatory key is missing instanciation should fail."""
        filter_config = {"filter_method": "disparity_denoiser", "sigma_color": 100.0}
        del filter_config[missing_key]

        with pytest.raises((MissKeyCheckerError, KeyError)):
            flt.AbstractFilter(cfg=filter_config)

    def test_check_conf_fails_when_there_is_wrong_values(self):
        """When there is a wrong type value instanciation should fail."""
        filter_config = {"filter_method": "disparity_denoiser", "sigma_color": 100}

        with pytest.raises((DictCheckerError, KeyError)):
            flt.AbstractFilter(cfg=filter_config)
