# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains tests of the pandora machine."""
import copy

import numpy as np
import pytest
import xarray as xr
from pandora import PandoraMachine


class TestFilterCheckConf:
    """Test filter_check_conf method."""

    @pytest.fixture()
    def left_monoband_image(self):
        """Return a 2D dataset builder of a given shape."""

        def inner(shape):
            return xr.Dataset(
                {
                    "im": (["row", "col"], np.empty(shape)),
                },
                coords={"row": np.arange(shape[0]), "col": np.arange(shape[1])},
            )

        return inner

    @pytest.fixture()
    def left_multiband_image(self):
        """Return a 3D dataset builder of a given shape."""

        def inner(shape):
            return xr.Dataset(
                {
                    "im": (["band_im", "row", "col"], np.empty((3, *shape))),
                },
                coords={"band_im": ["R", "G", "B"], "row": np.arange(shape[0]), "col": np.arange(shape[1])},
            )

        return inner

    @pytest.fixture()
    def pandora_machine_builder(self):
        """Return a pandora machine builder which expects an image shape."""

        def builder(image_shape, image_builder):
            machine = PandoraMachine()
            machine.left_img = image_builder(image_shape)
            return machine

        return builder

    @pytest.fixture()
    def user_pipeline(self):
        """Pipeline given by user."""
        user_cfg = {
            "filter": {"filter_method": "bilateral", "sigma_color": 4.0, "sigma_space": 6.0},
        }
        return user_cfg

    @pytest.mark.parametrize(
        "image_shape",
        [
            (3, 4),
            (10, 7),
        ],
    )
    @pytest.mark.parametrize(
        "image_builder_name",
        [
            "left_monoband_image",
            "left_multiband_image",
        ],
    )
    def test_filter_check_conf(self, request, image_builder_name, image_shape, pandora_machine_builder, user_pipeline):
        """Test image_shape is added to bilateralfilter config."""
        expected = copy.deepcopy(user_pipeline["filter"])
        expected["image_shape"] = image_shape
        pandora_machine = pandora_machine_builder(image_shape, request.getfixturevalue(image_builder_name))

        pandora_machine.filter_check_conf(user_pipeline, "filter")

        assert pandora_machine.pipeline_cfg["pipeline"]["filter"] == expected
