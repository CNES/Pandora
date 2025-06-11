# type:ignore
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
# pylint: disable=too-many-lines


"""
This module contains functions to test the disparity denoiser filter.
"""

import pytest
from json_checker import MissKeyCheckerError, DictCheckerError
import numpy as np
import xarray as xr

import pandora.constants as cst
import pandora.filter as flt


class TestDisparityDenoiser:
    """Test DisparityDenoiser"""

    @pytest.fixture(params=["monoband", "multiband"])
    def test_image(self, request, filter_cfg):
        """
        Initialization of a multiband and a monoband image
        """
        if request.param == "monoband":
            # initialize monoband data
            data = np.zeros((2, 2))
            data[:, :] = np.array(
                ([1, 1], [1, 3]),
                dtype=np.float64,
            )

            left = xr.Dataset(
                {"im": (["row", "col"], data)},
                coords={
                    "row": np.arange(data.shape[0]),
                    "col": np.arange(data.shape[1]),
                },
            )

            filter_cfg["band"] = None
        else:
            # Initialize multiband data
            data = np.zeros((3, 2, 2))
            data[0, :, :] = np.array(
                ([1, 1], [1, 3]),
                dtype=np.float64,
            )

            data[1, :, :] = np.array(
                ([2, 3], [8, 7]),
                dtype=np.float64,
            )

            data[2, :, :] = np.array(
                ([2, 3], [8, 8]),
                dtype=np.float64,
            )

            left = xr.Dataset(
                {"im": (["band_im", "row", "col"], data)},
                coords={
                    "band_im": ["red", "green", "blue"],
                    "row": np.arange(data.shape[1]),
                    "col": np.arange(data.shape[2]),
                },
            )

            filter_cfg["band"] = "red"

        return left, filter_cfg

    @pytest.fixture
    def filter_cfg(self):
        """
        Define the configuration
        """
        return {
            "filter_method": "disparity_denoiser",
            "filter_size": 11,
            "sigma_euclidian": 4.0,
            "sigma_color": 100.0,
            "sigma_planar": 12.0,
        }

    def test_disparity_denoiser_filter(self):
        """
        Instantiate a disparity_denoiser Filter.
        """
        return flt.AbstractFilter(cfg={"filter_method": "disparity_denoiser"})

    def test_check_conf_with_monoband(self, filter_cfg):
        """Test check conf with new values"""
        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        assert disparity_denoiser.cfg == filter_cfg

    def test_check_conf_with_new_values(self):
        """Test check conf with new values"""
        filter_config = {
            "filter_method": "disparity_denoiser",
            "filter_size": 9,
            "sigma_euclidian": 5.0,
            "sigma_color": 90.0,
            "sigma_planar": 10.0,
            "band": "red",
        }

        disparity_denoiser = flt.AbstractFilter(cfg=filter_config)

        assert disparity_denoiser.cfg == filter_config

    @pytest.mark.parametrize("missing_key", ["filter_method"])
    def test_check_conf_fails_when_is_missing_mandatory_key(self, missing_key, filter_cfg):
        """When a mandatory key is missing instanciation should fail."""
        del filter_cfg[missing_key]

        with pytest.raises((MissKeyCheckerError, KeyError)):
            flt.AbstractFilter(cfg=filter_cfg)

    def test_check_conf_fails_when_there_is_wrong_values(self, filter_cfg):
        """When there is a wrong type value instanciation should fail."""
        filter_cfg["sigma_color"] = 100

        with pytest.raises((DictCheckerError, KeyError)):
            flt.AbstractFilter(cfg=filter_cfg)

    def test_get_grad(self, filter_cfg):
        """Test the get grad function on a simple case"""
        filter_cfg["sigma_grad"] = 0.0

        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        disp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        dy, dx = disparity_denoiser.get_grad(disp)

        gt_y = np.array(
            [
                [(4 - 1) / 1, (5 - 2) / 1, (6 - 3) / 1],
                [(7 - 1) / 2, (8 - 2) / 2, (9 - 3) / 2],
                [(7 - 4) / 1, (8 - 5) / 1, (9 - 6) / 1],
            ]
        )

        gt_x = np.array(
            [
                [(2 - 1) / 1, (3 - 1) / 2, (3 - 2) / 1],
                [(5 - 4) / 1, (6 - 4) / 2, (6 - 5) / 1],
                [(8 - 7) / 1, (9 - 7) / 2, (9 - 8) / 1],
            ]
        )

        assert np.allclose(dx, gt_x, atol=1e-7)
        assert np.allclose(dy, gt_y, atol=1e-7)

    def test_sliding_window(self, filter_cfg):
        """
        Test the sliding_window method
        """

        # instantiate the filter
        filter_cfg["filter_size"] = 3

        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        disp = np.array([[1, 2], [4, 5]], dtype=float)

        disp = disp[np.newaxis, :, :]

        # gt_pad = np.array(
        #     [
        #         [5, 4, 5, 4],
        #         [2, 1, 2, 1],
        #         [5, 4, 5, 4],
        #         [2, 1, 2, 1],
        #     ],
        #     dtype=float,
        # )

        # calculate the truth value
        gt_window_view = np.array(
            [
                [
                    [[[5.0, 4.0, 5.0], [2.0, 1.0, 2.0], [5.0, 4.0, 5.0]]],
                    [[[4.0, 5.0, 4.0], [1.0, 2.0, 1.0], [4.0, 5.0, 4.0]]],
                ],
                [
                    [[[2.0, 1.0, 2.0], [5.0, 4.0, 5.0], [2.0, 1.0, 2.0]]],
                    [[[1.0, 2.0, 1.0], [4.0, 5.0, 4.0], [1.0, 2.0, 1.0]]],
                ],
            ],
        )

        window_view = disparity_denoiser.sliding_window(disp)

        assert np.allclose(window_view, gt_window_view, atol=1e-7)

    def test_disparity_and_color_dist(self, filter_cfg):
        """
        Test get_color_dist and get_disparity_dist
        """

        # instantiate the filter
        filter_cfg["filter_size"] = 3

        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        # disp = np.array([[1, 2], [4, 5]], dtype=float)

        disp_view = np.array(
            [
                [
                    [[[5.0, 4.0, 5.0], [2.0, 1.0, 2.0], [5.0, 4.0, 5.0]]],
                    [[[4.0, 5.0, 4.0], [1.0, 2.0, 1.0], [4.0, 5.0, 4.0]]],
                ],
                [
                    [[[2.0, 1.0, 2.0], [5.0, 4.0, 5.0], [2.0, 1.0, 2.0]]],
                    [[[1.0, 2.0, 1.0], [4.0, 5.0, 4.0], [1.0, 2.0, 1.0]]],
                ],
            ],
        )

        # calculate the truth value
        gt_disparity_dist = np.array(
            [
                [
                    [
                        [
                            [
                                disp_view[0, 0, 0, 0, 0] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 0, 1] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 0, 2] - disp_view[0, 0, 0, 1, 1],
                            ],
                            [
                                disp_view[0, 0, 0, 1, 0] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 1, 1] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 1, 2] - disp_view[0, 0, 0, 1, 1],
                            ],
                            [
                                disp_view[0, 0, 0, 2, 0] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 2, 1] - disp_view[0, 0, 0, 1, 1],
                                disp_view[0, 0, 0, 2, 2] - disp_view[0, 0, 0, 1, 1],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                disp_view[0, 1, 0, 0, 0] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 0, 1] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 0, 2] - disp_view[0, 1, 0, 1, 1],
                            ],
                            [
                                disp_view[0, 1, 0, 1, 0] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 1, 1] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 1, 2] - disp_view[0, 1, 0, 1, 1],
                            ],
                            [
                                disp_view[0, 1, 0, 2, 0] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 2, 1] - disp_view[0, 1, 0, 1, 1],
                                disp_view[0, 1, 0, 2, 2] - disp_view[0, 1, 0, 1, 1],
                            ],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [
                                disp_view[1, 0, 0, 0, 0] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 0, 1] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 0, 2] - disp_view[1, 0, 0, 1, 1],
                            ],
                            [
                                disp_view[1, 0, 0, 1, 0] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 1, 1] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 1, 2] - disp_view[1, 0, 0, 1, 1],
                            ],
                            [
                                disp_view[1, 0, 0, 2, 0] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 2, 1] - disp_view[1, 0, 0, 1, 1],
                                disp_view[1, 0, 0, 2, 2] - disp_view[1, 0, 0, 1, 1],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                disp_view[1, 1, 0, 0, 0] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 0, 1] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 0, 2] - disp_view[1, 1, 0, 1, 1],
                            ],
                            [
                                disp_view[1, 1, 0, 1, 0] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 1, 1] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 1, 2] - disp_view[1, 1, 0, 1, 1],
                            ],
                            [
                                disp_view[1, 1, 0, 2, 0] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 2, 1] - disp_view[1, 1, 0, 1, 1],
                                disp_view[1, 1, 0, 2, 2] - disp_view[1, 1, 0, 1, 1],
                            ],
                        ]
                    ],
                ],
            ]
        )

        disparity_dist = disparity_denoiser.get_disparity_dist(disp_view)

        assert np.allclose(disparity_dist, gt_disparity_dist, atol=1e-7)

        # it will be the same for the color dist
        color_dist = disparity_denoiser.get_color_dist(disp_view)

        assert np.allclose(color_dist, gt_disparity_dist, atol=1e-7)

    def test_planar_dist(self, filter_cfg):
        """
        Test the function get_planar_dist
        """

        # instantiate the filter
        filter_cfg["filter_size"] = 3

        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        # disp = np.array([[1, 2], [4, 5]], dtype=float)

        disp_view = np.array(
            [
                [
                    [[[5.0, 4.0, 5.0], [2.0, 1.0, 2.0], [5.0, 4.0, 5.0]]],
                    [[[4.0, 5.0, 4.0], [1.0, 2.0, 1.0], [4.0, 5.0, 4.0]]],
                ],
                [
                    [[[2.0, 1.0, 2.0], [5.0, 4.0, 5.0], [2.0, 1.0, 2.0]]],
                    [[[1.0, 2.0, 1.0], [4.0, 5.0, 4.0], [1.0, 2.0, 1.0]]],
                ],
            ],
        )

        # Use those functions to get the disp grad view as they have been tested before
        # disp_grad = disparity_denoiser.get_grad(disp.squeeze())
        # disp_grad_view = disparity_denoiser.sliding_window(disp_grad)

        disp_grad_view = np.array(
            [
                [
                    [
                        [
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                        ],
                        [
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                        ],
                    ],
                    [
                        [
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                        ],
                        [
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                        ],
                        [
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                        ],
                    ],
                    [
                        [
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                            [0.18689481, 0.18689481, 0.18689481],
                        ],
                        [
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                            [0.06229827, 0.06229827, 0.06229827],
                        ],
                    ],
                ],
            ],
        )

        # The win coords is a mesh grid using the window size of 3
        win_coords = np.array([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])

        # step to calculate the gt_dist
        arr = np.array(
            [
                [
                    [
                        [
                            [
                                0.18689481 * win_coords[0, 0, 0],
                                0.18689481 * win_coords[0, 0, 1],
                                0.18689481 * win_coords[0, 0, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 1, 0],
                                0.18689481 * win_coords[0, 1, 1],
                                0.18689481 * win_coords[0, 1, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 2, 0],
                                0.18689481 * win_coords[0, 2, 1],
                                0.18689481 * win_coords[0, 2, 2],
                            ],
                        ],
                        [
                            [
                                0.06229827 * win_coords[1, 0, 0],
                                0.06229827 * win_coords[1, 0, 1],
                                0.06229827 * win_coords[1, 0, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 1, 0],
                                0.06229827 * win_coords[1, 1, 1],
                                0.06229827 * win_coords[1, 1, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 2, 0],
                                0.06229827 * win_coords[1, 2, 1],
                                0.06229827 * win_coords[1, 2, 2],
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                0.18689481 * win_coords[0, 0, 0],
                                0.18689481 * win_coords[0, 0, 1],
                                0.18689481 * win_coords[0, 0, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 1, 0],
                                0.18689481 * win_coords[0, 1, 1],
                                0.18689481 * win_coords[0, 1, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 2, 0],
                                0.18689481 * win_coords[0, 2, 1],
                                0.18689481 * win_coords[0, 2, 2],
                            ],
                        ],
                        [
                            [
                                0.06229827 * win_coords[1, 0, 0],
                                0.06229827 * win_coords[1, 0, 1],
                                0.06229827 * win_coords[1, 0, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 1, 0],
                                0.06229827 * win_coords[1, 1, 1],
                                0.06229827 * win_coords[1, 1, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 2, 0],
                                0.06229827 * win_coords[1, 2, 1],
                                0.06229827 * win_coords[1, 2, 2],
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [
                                0.18689481 * win_coords[0, 0, 0],
                                0.18689481 * win_coords[0, 0, 1],
                                0.18689481 * win_coords[0, 0, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 1, 0],
                                0.18689481 * win_coords[0, 1, 1],
                                0.18689481 * win_coords[0, 1, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 2, 0],
                                0.18689481 * win_coords[0, 2, 1],
                                0.18689481 * win_coords[0, 2, 2],
                            ],
                        ],
                        [
                            [
                                0.06229827 * win_coords[1, 0, 0],
                                0.06229827 * win_coords[1, 0, 1],
                                0.06229827 * win_coords[1, 0, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 1, 0],
                                0.06229827 * win_coords[1, 1, 1],
                                0.06229827 * win_coords[1, 1, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 2, 0],
                                0.06229827 * win_coords[1, 2, 1],
                                0.06229827 * win_coords[1, 2, 2],
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                0.18689481 * win_coords[0, 0, 0],
                                0.18689481 * win_coords[0, 0, 1],
                                0.18689481 * win_coords[0, 0, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 1, 0],
                                0.18689481 * win_coords[0, 1, 1],
                                0.18689481 * win_coords[0, 1, 2],
                            ],
                            [
                                0.18689481 * win_coords[0, 2, 0],
                                0.18689481 * win_coords[0, 2, 1],
                                0.18689481 * win_coords[0, 2, 2],
                            ],
                        ],
                        [
                            [
                                0.06229827 * win_coords[1, 0, 0],
                                0.06229827 * win_coords[1, 0, 1],
                                0.06229827 * win_coords[1, 0, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 1, 0],
                                0.06229827 * win_coords[1, 1, 1],
                                0.06229827 * win_coords[1, 1, 2],
                            ],
                            [
                                0.06229827 * win_coords[1, 2, 0],
                                0.06229827 * win_coords[1, 2, 1],
                                0.06229827 * win_coords[1, 2, 2],
                            ],
                        ],
                    ],
                ],
            ],
        )

        sum_res = np.array(
            [
                [
                    [
                        [
                            [
                                arr[0, 0, 0, 0, 0] + arr[0, 0, 1, 0, 0],
                                arr[0, 0, 0, 0, 1] + arr[0, 0, 1, 0, 1],
                                arr[0, 0, 0, 0, 2] + arr[0, 0, 1, 0, 2],
                            ],
                            [
                                arr[0, 0, 0, 1, 0] + arr[0, 0, 1, 1, 0],
                                arr[0, 0, 0, 1, 1] + arr[0, 0, 1, 1, 1],
                                arr[0, 0, 0, 1, 2] + arr[0, 0, 1, 1, 2],
                            ],
                            [
                                arr[0, 0, 0, 2, 0] + arr[0, 0, 1, 2, 0],
                                arr[0, 0, 0, 2, 1] + arr[0, 0, 1, 2, 1],
                                arr[0, 0, 0, 2, 2] + arr[0, 0, 1, 2, 2],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                arr[0, 1, 0, 0, 0] + arr[0, 1, 1, 0, 0],
                                arr[0, 1, 0, 0, 1] + arr[0, 1, 1, 0, 1],
                                arr[0, 1, 0, 0, 2] + arr[0, 1, 1, 0, 2],
                            ],
                            [
                                arr[0, 1, 0, 1, 0] + arr[0, 1, 1, 1, 0],
                                arr[0, 1, 0, 1, 1] + arr[0, 1, 1, 1, 1],
                                arr[0, 1, 0, 1, 2] + arr[0, 1, 1, 1, 2],
                            ],
                            [
                                arr[0, 1, 0, 2, 0] + arr[0, 1, 1, 2, 0],
                                arr[0, 1, 0, 2, 1] + arr[0, 1, 1, 2, 1],
                                arr[0, 1, 0, 2, 2] + arr[0, 1, 1, 2, 2],
                            ],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [
                                arr[1, 0, 0, 0, 0] + arr[1, 0, 1, 0, 0],
                                arr[1, 0, 0, 0, 1] + arr[1, 0, 1, 0, 1],
                                arr[1, 0, 0, 0, 2] + arr[1, 0, 1, 0, 2],
                            ],
                            [
                                arr[1, 0, 0, 1, 0] + arr[1, 0, 1, 1, 0],
                                arr[1, 0, 0, 1, 1] + arr[1, 0, 1, 1, 1],
                                arr[1, 0, 0, 1, 2] + arr[1, 0, 1, 1, 2],
                            ],
                            [
                                arr[1, 0, 0, 2, 0] + arr[1, 0, 1, 2, 0],
                                arr[1, 0, 0, 2, 1] + arr[1, 0, 1, 2, 1],
                                arr[1, 0, 0, 2, 2] + arr[1, 0, 1, 2, 2],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                arr[1, 1, 0, 0, 0] + arr[1, 1, 1, 0, 0],
                                arr[1, 1, 0, 0, 1] + arr[1, 1, 1, 0, 1],
                                arr[1, 1, 0, 0, 2] + arr[1, 1, 1, 0, 2],
                            ],
                            [
                                arr[1, 1, 0, 1, 0] + arr[1, 1, 1, 1, 0],
                                arr[1, 1, 0, 1, 1] + arr[1, 1, 1, 1, 1],
                                arr[1, 1, 0, 1, 2] + arr[1, 1, 1, 1, 2],
                            ],
                            [
                                arr[1, 1, 0, 2, 0] + arr[1, 1, 1, 2, 0],
                                arr[1, 1, 0, 2, 1] + arr[1, 1, 1, 2, 1],
                                arr[1, 1, 0, 2, 2] + arr[1, 1, 1, 2, 2],
                            ],
                        ]
                    ],
                ],
            ],
        )

        gt_dist = disp_view - sum_res

        # case centered_plane at true
        offset = np.mean(gt_dist, axis=(-2, -1), keepdims=True)

        gt_planar_dist = gt_dist - offset
        planar_dist = disparity_denoiser.get_planar_dist(disp_view, disp_grad_view, centered_plane=True)

        assert np.allclose(gt_planar_dist, planar_dist, atol=1e-7)

        # case centered_plane at false, the offset will be the center of each submatrix of the disp_view
        offset = [[[[[1.0]]], [[[2.0]]]], [[[[4.0]]], [[[5.0]]]]]

        gt_planar_dist = gt_dist - offset
        planar_dist = disparity_denoiser.get_planar_dist(disp_view, disp_grad_view)

        assert np.allclose(planar_dist, gt_planar_dist, atol=1e-7)

    def test_with_valid_pixel_multiband_and_monoband(self, test_image):
        """
        Test the disparity denoiser method on valid pixels and multiband and monoband image.
        disparity denoiser filter is only applied on valid pixels.
        """
        left, user_cfg = test_image
        user_cfg["filter_size"] = 3

        disparity_denoiser = flt.AbstractFilter(cfg=user_cfg)

        # create the disparity map
        disp = np.array([[1, 2], [4, 5]], dtype=float)

        # validity mask
        valid = np.array(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=np.uint16,
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(2), "col": np.arange(2)},
        )

        # The win coords is a meshgrid using the size of the window 3 (from -1 to 1)
        win_coords = np.array([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])

        # The euclidian dist and the weights are the only variables which have not been tested
        gt_euclidian_dist = np.array(
            [
                [
                    [
                        [
                            [
                                np.sqrt(win_coords[0, 0, 0] ** 2 + win_coords[1, 0, 0] ** 2),
                                np.sqrt(win_coords[0, 0, 1] ** 2 + win_coords[1, 0, 1] ** 2),
                                np.sqrt(win_coords[0, 0, 2] ** 2 + win_coords[1, 0, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 1, 0] ** 2 + win_coords[1, 1, 0] ** 2),
                                np.sqrt(win_coords[0, 1, 1] ** 2 + win_coords[1, 1, 1] ** 2),
                                np.sqrt(win_coords[0, 1, 2] ** 2 + win_coords[1, 1, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 2, 0] ** 2 + win_coords[1, 2, 0] ** 2),
                                np.sqrt(win_coords[0, 2, 1] ** 2 + win_coords[1, 2, 1] ** 2),
                                np.sqrt(win_coords[0, 2, 2] ** 2 + win_coords[1, 2, 2] ** 2),
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                np.sqrt(win_coords[0, 0, 0] ** 2 + win_coords[1, 0, 0] ** 2),
                                np.sqrt(win_coords[0, 0, 1] ** 2 + win_coords[1, 0, 1] ** 2),
                                np.sqrt(win_coords[0, 0, 2] ** 2 + win_coords[1, 0, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 1, 0] ** 2 + win_coords[1, 1, 0] ** 2),
                                np.sqrt(win_coords[0, 1, 1] ** 2 + win_coords[1, 1, 1] ** 2),
                                np.sqrt(win_coords[0, 1, 2] ** 2 + win_coords[1, 1, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 2, 0] ** 2 + win_coords[1, 2, 0] ** 2),
                                np.sqrt(win_coords[0, 2, 1] ** 2 + win_coords[1, 2, 1] ** 2),
                                np.sqrt(win_coords[0, 2, 2] ** 2 + win_coords[1, 2, 2] ** 2),
                            ],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [
                                np.sqrt(win_coords[0, 0, 0] ** 2 + win_coords[1, 0, 0] ** 2),
                                np.sqrt(win_coords[0, 0, 1] ** 2 + win_coords[1, 0, 1] ** 2),
                                np.sqrt(win_coords[0, 0, 2] ** 2 + win_coords[1, 0, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 1, 0] ** 2 + win_coords[1, 1, 0] ** 2),
                                np.sqrt(win_coords[0, 1, 1] ** 2 + win_coords[1, 1, 1] ** 2),
                                np.sqrt(win_coords[0, 1, 2] ** 2 + win_coords[1, 1, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 2, 0] ** 2 + win_coords[1, 2, 0] ** 2),
                                np.sqrt(win_coords[0, 2, 1] ** 2 + win_coords[1, 2, 1] ** 2),
                                np.sqrt(win_coords[0, 2, 2] ** 2 + win_coords[1, 2, 2] ** 2),
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                np.sqrt(win_coords[0, 0, 0] ** 2 + win_coords[1, 0, 0] ** 2),
                                np.sqrt(win_coords[0, 0, 1] ** 2 + win_coords[1, 0, 1] ** 2),
                                np.sqrt(win_coords[0, 0, 2] ** 2 + win_coords[1, 0, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 1, 0] ** 2 + win_coords[1, 1, 0] ** 2),
                                np.sqrt(win_coords[0, 1, 1] ** 2 + win_coords[1, 1, 1] ** 2),
                                np.sqrt(win_coords[0, 1, 2] ** 2 + win_coords[1, 1, 2] ** 2),
                            ],
                            [
                                np.sqrt(win_coords[0, 2, 0] ** 2 + win_coords[1, 2, 0] ** 2),
                                np.sqrt(win_coords[0, 2, 1] ** 2 + win_coords[1, 2, 1] ** 2),
                                np.sqrt(win_coords[0, 2, 2] ** 2 + win_coords[1, 2, 2] ** 2),
                            ],
                        ]
                    ],
                ],
            ],
        )

        # Got those result from the function of disparity denoiser that have been tested before
        clr_dist = np.array(
            [
                [
                    [[[2.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]],
                    [[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
                ],
                [
                    [[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 0.0]]],
                    [[[-2.0, -2.0, -2.0], [-2.0, 0.0, -2.0], [-2.0, -2.0, -2.0]]],
                ],
            ],
        )

        planar_dist_centered = np.array(
            [
                [
                    [
                        [
                            [1.58252641, 0.52022814, 1.45792987],
                            [-1.6043684, -2.66666667, -1.72896494],
                            [1.20873679, 0.14643852, 1.08414026],
                        ]
                    ],
                    [
                        [
                            [0.91585974, 1.85356148, 0.79126321],
                            [-2.27103506, -1.33333333, -2.3956316],
                            [0.54207013, 1.47977186, 0.41747359],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [-0.41747359, -1.47977186, -0.54207013],
                            [2.3956316, 1.33333333, 2.27103506],
                            [-0.79126321, -1.85356148, -0.91585974],
                        ]
                    ],
                    [
                        [
                            [-1.08414026, -0.14643852, -1.20873679],
                            [1.72896494, 2.66666667, 1.6043684],
                            [-1.45792987, -0.52022814, -1.58252641],
                        ]
                    ],
                ],
            ]
        )

        planar_dist = np.array(
            [
                [
                    [
                        [
                            [4.24919308, 3.18689481, 4.12459654],
                            [1.06229827, 0.0, 0.93770173],
                            [3.87540346, 2.81310519, 3.75080692],
                        ]
                    ],
                    [
                        [
                            [2.24919308, 3.18689481, 2.12459654],
                            [-0.93770173, 0.0, -1.06229827],
                            [1.87540346, 2.81310519, 1.75080692],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [-1.75080692, -2.81310519, -1.87540346],
                            [1.06229827, 0.0, 0.93770173],
                            [-2.12459654, -3.18689481, -2.24919308],
                        ]
                    ],
                    [
                        [
                            [-3.75080692, -2.81310519, -3.87540346],
                            [-0.93770173, 0.0, -1.06229827],
                            [-4.12459654, -3.18689481, -4.24919308],
                        ]
                    ],
                ],
            ]
        )
        # Calculate the weights
        gt_weights = (
            1
            * np.exp(-np.power(gt_euclidian_dist / 4.0, 2.0) / 2.0)
            * np.exp(-np.power(clr_dist / 100.0, 2.0) / 2.0)
            * np.exp(-np.power(planar_dist_centered / 12.0, 2.0) / 2.0)
        )

        # Calculate the truth value
        gt = disparity_denoiser.bilateral_filter(disp[None, ...], planar_dist, gt_weights)

        # Calculate the disparity denoiser value
        disparity_denoiser.filter_disparity(disp_dataset, left)

        np.testing.assert_allclose(gt[0], disp_dataset["disparity_map"].data, rtol=1e-07)

        # instantiate the filter with monoband
        user_cfg = {"filter_method": "disparity_denoiser", "filter_size": 3}

        disparity_denoiser = flt.AbstractFilter(cfg=user_cfg)

    @staticmethod
    def test_with_invalid_center(filter_cfg):
        """
        Test the disparity denoiser method with center pixel invalid.
        disparity denoiser filter is only applied on valid pixels.

        """

        # instantiate the filter
        disparity_denoiser = flt.AbstractFilter(cfg=filter_cfg)

        # create the disparity map
        disp = np.array(
            [[2, 4, 8, 5, 6], [7, 82, 3, 33, 4], [4, 8, 21, 13, 4], [3, 2, 8, 1, 3], [3, 6, 2, 3, 2]], dtype=np.float32
        )

        # validity mask
        validity_mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, cst.PANDORA_MSK_PIXEL_INVALID, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], validity_mask)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

        # Initialize multiband data
        data = np.zeros((3, 5, 5))
        data[0, :, :] = np.array(
            (
                [1, 1, 1, 3, 1],
                [1, 3, 2, 5, 4],
                [2, 1, 0, 1, 4],
                [1, 5, 4, 3, 8],
                [1, 5, 4, 3, 9],
            ),
            dtype=np.float64,
        )

        data[1, :, :] = np.array(
            (
                [2, 3, 4, 6, 8],
                [8, 7, 0, 4, 7],
                [4, 9, 1, 5, 1],
                [6, 5, 2, 1, 4],
                [1, 5, 4, 3, 2],
            ),
            dtype=np.float64,
        )

        data[2, :, :] = np.array(
            (
                [2, 3, 4, 6, 3],
                [8, 8, 0, 4, 1],
                [4, 9, 1, 5, 2],
                [6, 5, 4, 1, 1],
                [1, 5, 4, 3, 3],
            ),
            dtype=np.float64,
        )

        left = xr.Dataset(
            {"im": (["band_im", "row", "col"], data)},
            coords={
                "band_im": ["red", "green", "blue"],
                "row": np.arange(data.shape[1]),
                "col": np.arange(data.shape[2]),
            },
        )

        # Apply bilateral filter to the disparity map
        disparity_denoiser.filter_disparity(disp_dataset, left)

        # Filtered disparity center must be the same as input since center pixel was invalid
        np.testing.assert_allclose(disp[2, 2], disp_dataset["disparity_map"][2, 2], rtol=1e-07)
