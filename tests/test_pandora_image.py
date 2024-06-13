# pylint:disable=too-many-lines
# type:ignore
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
This module contains functions to test all the methods in img_tools module.
"""

# pylint: disable=redefined-outer-name

import copy
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
import xarray as xr

import pandora
from pandora import img_tools
from pandora.img_tools import rasterio_open, create_dataset_from_inputs
from tests import common


@pytest.fixture()
def monoband_image():
    """Create monoband image."""
    data = np.array(
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
    )

    return xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
    )


@pytest.fixture()
def multiband_image():
    """Create multiband image"""
    return common.matching_cost_tests_multiband_setup()[0]


@pytest.mark.parametrize(
    ["window_size", "expected"],
    [
        pytest.param(
            3,
            np.array(
                (
                    [0b000000000, 0b000000001, 0b000001011, 0b000000110],
                    [0b000000000, 0b000001000, 0b000000000, 0b000100000],
                    [0b000000000, 0b001000000, 0b011000000, 0b110000000],
                )
            ),
            id="Window size of 3",
        ),
        pytest.param(5, np.array(([[0b0000000001000110000000000, 0b0]])), id="Window size of 5"),
    ],
)
class TestSensusTransform:
    """Test census_transform function."""

    def test_monoband(self, monoband_image, window_size, expected):
        """
        Test the census transform method
        """
        # Computes the census transform for the image with window_size
        census_transform = img_tools.census_transform(monoband_image, window_size)
        # Check if the census_transform is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census_transform["im"].data, expected)

    def test_multiband(self, multiband_image, window_size, expected):
        """
        Test the census transform method for multiband image
        """
        # Computes the census transform for the image self.img_multiband with window size 5
        census_transform = img_tools.census_transform(multiband_image, window_size, "r")
        # Check if the census_transform is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census_transform["im"].data, expected)


@pytest.mark.parametrize(
    ["window_size", "expected"],
    [
        pytest.param(
            3,
            np.array(
                (
                    [1.0, 12 / 9.0, 15 / 9.0, 15 / 9.0],
                    [1.0, 12 / 9.0, 15 / 9.0, 15 / 9.0],
                    [1.0, 12 / 9.0, 14.0 / 9, 14.0 / 9],
                )
            ),
            id="Window size of 3",
        ),
        pytest.param(5, np.array(([[31 / 25.0, 31 / 25.0]])), id="Window size of 5"),
    ],
)
class TestComputeMeanMaster:
    """Test compute_mean_raster function."""

    def test_monoband_raster(self, monoband_image, window_size, expected):
        """
        Test the method compute_mean_raster

        """
        # Computes the mean raster for the image with window_size
        mean_r = img_tools.compute_mean_raster(monoband_image, window_size)
        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(mean_r, expected)

    def test_multiband_raster(self, multiband_image, window_size, expected):
        """
        Test the method compute_mean_raster

        """
        # Computes the mean raster for the image with window_size
        mean_r = img_tools.compute_mean_raster(multiband_image, window_size, "r")
        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(mean_r, expected)


@pytest.mark.parametrize(["row", "col", "window_size", "expected"], [[1, 1, 3, 1.0], [2, 2, 5, np.float32(31 / 25.0)]])
def test_compute_mean_patch(monoband_image, row, col, window_size, expected):
    """
    Test the method compute_mean_patch

    """
    # Computes the mean for the image with window_size centered on col, row
    mean = img_tools.compute_mean_patch(monoband_image, row, col, window_size)

    assert mean == expected


@pytest.mark.parametrize(
    ["row", "col", "expected"],
    [
        pytest.param(0, 0, True, id="Is inside"),
        pytest.param(-1, 0, False, id="Is outside"),
        pytest.param(0, 6, False, id="Is outside with row, col inverted"),
    ],
)
def test_check_inside_image(monoband_image, row, col, expected):
    """
    Test the method check_inside_image

    """
    assert img_tools.check_inside_image(monoband_image, row, col) is expected


class TestStdRaster:
    """Test compute_std_raster function."""

    def test_monoband_with_window_size_of_3(self, monoband_image):
        """
        Test the method compute_std_raster

        """
        # standard deviation raster ground truth for the image self.img with window size 3
        std_ground_truth = np.array(
            (
                [
                    0.0,
                    np.std(monoband_image["im"][:3, 1:4]),
                    np.std(monoband_image["im"][:3, 2:5]),
                    np.std(monoband_image["im"][:3, 3:]),
                ],
                [
                    0.0,
                    np.std(monoband_image["im"][1:4, 1:4]),
                    np.std(monoband_image["im"][1:4, 2:5]),
                    np.std(monoband_image["im"][1:4, 3:]),
                ],
                [
                    0.0,
                    np.std(monoband_image["im"][2:5, 1:4]),
                    np.std(monoband_image["im"][2:5, 2:5]),
                    np.std(monoband_image["im"][2:5, 3:]),
                ],
            )
        )
        # Computes the standard deviation raster for the image with window size 3
        std_r = img_tools.compute_std_raster(monoband_image, 3)
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)

    def test_monoband_with_window_size_of_5(self, monoband_image):
        """Test monoband with window size of 5."""
        # standard deviation raster ground truth for the image with window size 5
        std_ground_truth = np.array(([[np.std(monoband_image["im"][:, :5]), np.std(monoband_image["im"][:, 1:])]]))
        # Computes the standard deviation raster for the image with window size 5
        std_r = img_tools.compute_std_raster(monoband_image, 5)
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)

    def test_multiband_with_window_size_of_3(self, multiband_image):
        """Test multiband with window size of 3."""
        # standard deviation raster ground truth for the image self.img with window size 3
        std_ground_truth = np.array(
            (
                [
                    0.0,
                    np.std(multiband_image["im"][0, :3, 1:4]),
                    np.std(multiband_image["im"][0, :3, 2:5]),
                    np.std(multiband_image["im"][0, :3, 3:]),
                ],
                [
                    0.0,
                    np.std(multiband_image["im"][0, 1:4, 1:4]),
                    np.std(multiband_image["im"][0, 1:4, 2:5]),
                    np.std(multiband_image["im"][0, 1:4, 3:]),
                ],
                [
                    0.0,
                    np.std(multiband_image["im"][0, 2:5, 1:4]),
                    np.std(multiband_image["im"][0, 2:5, 2:5]),
                    np.std(multiband_image["im"][0, 2:5, 3:]),
                ],
            )
        )
        # Computes the standard deviation raster for the image self.img with window size 3
        std_r = img_tools.compute_std_raster(multiband_image, 3, "r")
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)

    def test_multiband_with_window_size_of_5(self, multiband_image):
        """Test multiband with window size of 5."""
        # standard deviation raster ground truth for the image self.img with window size 5
        std_ground_truth = np.array(
            ([[np.std(multiband_image["im"][0, :, :5]), np.std(multiband_image["im"][0, :, 1:])]])
        )
        # Computes the standard deviation raster for the image self.img with window size 5
        std_r = img_tools.compute_std_raster(multiband_image, 5, "r")
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)


class TestGetWindow:
    """Test get_window function."""

    @pytest.fixture()
    def default_image_shape(self):
        """
        Create a fake image to test roi configuration
        """
        imarray = np.array(
            (
                [0, 1, 2, 5, 1, 3, 6, 4, 9, 7, 8],
                [5, 1, 2, 7, 1, 4, 7, 8, 5, 8, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [4, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
                [2, 3, 5, 0, 1, 5, 9, 2, 8, 6, 7],
                [1, 2, 4, 5, 2, 6, 7, 7, 3, 7, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [4, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
            )
        )
        return imarray.shape

    @staticmethod
    def test_roi_inside(default_image_shape):
        """
        Test the get_window method when the config has a roi inside the image

        """
        img_height, img_width = default_image_shape

        # Roi
        roi = {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]}

        # get_window
        window = img_tools.get_window(roi, img_width, img_height)
        assert Window(col_off=1, row_off=1, width=7, height=7) == window

    @pytest.mark.parametrize(
        ["roi", "expected"],
        [
            pytest.param(
                {"col": {"first": 0, "last": 2}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                Window(col_off=0, row_off=1, width=5, height=7),
                id="Overlap on left side",
            ),
            pytest.param(
                {"col": {"first": 10, "last": 12}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                Window(col_off=8, row_off=1, width=3, height=7),
                id="Overlap on right side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": -1, "last": 5}, "margins": [2, 2, 2, 2]},
                Window(col_off=1, row_off=0, width=7, height=8),
                id="Overlap on up side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 9, "last": 11}, "margins": [2, 2, 2, 2]},
                Window(col_off=1, row_off=7, width=7, height=1),
                id="Overlap on down side",
            ),
        ],
    )
    def test_overlap_roi(self, default_image_shape, roi, expected):
        """
        Test the get_window method when the config has a roi overlaped with image

        """
        img_height, img_width = default_image_shape

        assert img_tools.get_window(roi, img_width, img_height) == expected

    @pytest.mark.parametrize(
        ["roi"],
        [
            pytest.param(
                {"col": {"first": -10, "last": -12}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                id="Outside on left side",
            ),
            pytest.param(
                {"col": {"first": 100, "last": 120}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                id="Outside on right side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": -6, "last": -5}, "margins": [2, 2, 2, 2]},
                id="Outside on up side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 11, "last": 111}, "margins": [2, 2, 2, 2]},
                id="Outside on down side",
            ),
        ],
    )
    def test_fails_when_roi_is_outside_image(self, default_image_shape, roi):
        """
        Test the get_window method when the config has a roi outside an image

        """
        img_height, img_width = default_image_shape

        with pytest.raises(ValueError, match="Roi specified is outside the image"):
            img_tools.get_window(roi, img_width, img_height)


class TestCreateDatasetFromInputs:
    """Test create_dataset_from_inputs function."""

    @pytest.fixture()
    def default_cfg(self):
        """Get default configuration."""
        return pandora.check_configuration.default_short_configuration

    @staticmethod
    def test_create_dataset_from_inputs(default_cfg):
        """
        Test the method create_dataset_from_inputs

        """
        # left_img = array([[ -9999.,  1.,  2.,  3.,  -9999.],
        #                  [ 5.,  6.,  7.,  8.,  9.],
        #                  [ -9999.,  -9999., 23.,  5.,  6.],
        #                  [12.,  5.,  6.,  3.,  -9999.]], dtype=float32)

        # Convention 0 is a valid pixel, everything else is considered invalid
        # mask_left = array([[  0,   0,   1,   2,   0],
        #                   [  0,   0,   0,   0,   1],
        #                   [  3,   5,   0,   0,   1],
        #                   [  0,   0, 565,   0,   0]])

        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/image/left_img.tif",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "mask": "tests/image/mask_left.tif",
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # Mask ground truth
        mask_gt = np.array([[1, 0, 2, 2, 1], [0, 0, 0, 0, 2], [1, 1, 0, 0, 2], [0, 0, 2, 0, 1]])

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left["msk"].data, mask_gt)

        left_img = np.array(
            [
                [-9999.0, 1.0, 2.0, 3.0, -9999.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [-9999.0, -9999.0, 23.0, 5.0, 6.0],
                [12.0, 5.0, 6.0, 3.0, -9999.0],
            ],
            dtype=np.float32,
        )

        # Check the image
        np.testing.assert_array_equal(dst_left["im"].data, left_img)

    @staticmethod
    def test_with_nan():
        """
        Test the method create_dataset_from_inputs

        """
        # left_img = array([[ NaN,  1.,  2.,  3.,  NaN],
        #                  [ 5.,  6.,  7.,  8.,  9.],
        #                  [ NaN,  0., 23.,  5.,  6.],
        #                  [12.,  5.,  6.,  3.,  NaN]], dtype=float32)

        # Convention 0 is a valid pixel, everything else is considered invalid
        # mask_left = array([[  0,   0,   1,   2,   0],
        #                   [  0,   0,   0,   0,   1],
        #                   [  3,   5,   0,   0,   1],
        #                   [  0,   0, 565,   0,   0]])

        # Computes the dataset image and use nan as no data,not cfg value
        input_config = {
            "left": {
                "img": "tests/image/left_img_nan.tif",
                "nodata": np.nan,
                "mask": "tests/image/mask_left.tif",
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # Mask ground truth
        mask_gt = np.array([[1, 0, 2, 2, 1], [0, 0, 0, 0, 2], [1, 1, 0, 0, 2], [0, 0, 2, 0, 1]])

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left["msk"].data, mask_gt)

        left_img = np.array(
            [
                [-9999.0, 1.0, 2.0, 3.0, -9999.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [-9999.0, -9999.0, 23.0, 5.0, 6.0],
                [12.0, 5.0, 6.0, 3.0, -9999.0],
            ],
            dtype=np.float32,
        )

        # Check the image
        np.testing.assert_array_equal(dst_left["im"].data, left_img)

    @staticmethod
    def test_with_classif(default_cfg):
        """
        Test the method create_dataset_from_inputs for the classif

        """
        # Computes the dataset image
        # The classes present in left_classif are "cornfields", "olive tree", "forest"
        input_config = {
            "left": {
                "img": "tests/pandora/left.png",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "classif": "tests/pandora/left_classif.tif",
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # The classes present in left_classif are "cornfields", "olive tree", "forest"
        gt_classes = ["cornfields", "olive tree", "forest"]

        # Check if the classes names are correctly set on the dataset
        np.testing.assert_array_equal(list(dst_left.band_classif.data), gt_classes)

    @staticmethod
    def test_rgb_image_with_classif(default_cfg):
        """
        Test the method create_dataset_from_inputs for the multiband image and classif

        """
        # Computes the dataset image
        # The classes present in left_classif are "cornfields", "olive tree", "forest"
        input_config = {
            "left": {
                "img": "tests/pandora/left_rgb.tif",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "classif": "tests/pandora/left_classif.tif",
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # The bands present in left_rgb are "r", "g", "b"
        gt_bands = ["r", "g", "b"]
        # Check if the classes names are correctly set on the dataset
        np.testing.assert_array_equal(list(dst_left.band_im.data), gt_bands)

        # The classes present in left_classif are "cornfields", "olive tree", "forest"
        gt_classes = ["cornfields", "olive tree", "forest"]
        # Check if the classes names are correctly set on the dataset
        np.testing.assert_array_equal(list(dst_left.band_classif.data), gt_classes)

    @staticmethod
    def test_rgb_image_with_mask():
        """
        Test the method create_dataset_from_inputs for the multiband image and mask

        """
        # img_left = array([[[ 181.   182.   178.]
        #                    [ 169.   175.   176.]
        #                    [ 176.   166.   162.]]
        #
        #                   [[  49.    44.    44.]
        #                    [  37.    34.    44.]
        #                    [  77.    68.    48.]]
        #
        #                   [[  49.    46.    43.]
        #                    [  38.    37.    41.]
        #                    [ 109.    75.    39.]]]
        # mask_left = array([[0  0  1],
        #                   [ 0  0  0],
        #                   [ 3  5  0]])

        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/pandora/left_rgb.tif",
                "nodata": 37.0,
                "mask": "tests/image/mask_left.tif",
                "disp": [-60, 0],
            }
        }

        roi = {"col": {"first": 0, "last": 2}, "row": {"first": 0, "last": 2}, "margins": [0, 0, 0, 0]}
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=roi)

        # Mask ground truth
        mask_gt = np.array(
            [[0, 0, 2], [1, 1, 0], [2, 2, 0]],
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left["msk"].data, mask_gt)
        assert dst_left["msk"].shape == (3, 3)
        assert dst_left["im"].shape == (3, 3, 3)

    @staticmethod
    def test_with_segm(default_cfg):
        """
        Test the method create_dataset_from_inputs for the segmentation

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/image/left_img.tif",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "segm": "tests/image/mask_left.tif",
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # Segmentation ground truth
        segm_gt = np.array(
            [[0, 0, 1, 2, 0], [0, 0, 0, 0, 1], [3, 5, 0, 0, 1], [0, 0, 255, 0, 0]],
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left["segm"].data, segm_gt)

    @staticmethod
    def test_with_geotransform(default_cfg):
        """
        Test the method create_dataset_from_inputs with an image with geotransform

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/pandora/left.png",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        gt_crs = rasterio.crs.CRS.from_epsg(32631)
        gt_transform = rasterio.Affine(0.5, 0.0, 573083.5, 0.0, -0.5, 4825333.5)

        # Check if the CRS and Transform are correctly read
        np.testing.assert_array_equal(gt_crs, dst_left.attrs["crs"])
        np.testing.assert_array_equal(gt_transform, dst_left.attrs["transform"])

    @staticmethod
    def test_without_geotransform(default_cfg):
        """
        Test the method create_dataset_from_inputs with an image without geotransform

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/image/left_img.tif",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "disp": [-60, 0],
            }
        }
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        gt_crs = None
        gt_transform = None

        # Check if the CRS and Transform are correctly set to None
        np.testing.assert_array_equal(gt_crs, dst_left.attrs["crs"])
        np.testing.assert_array_equal(gt_transform, dst_left.attrs["transform"])

    @staticmethod
    @pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
    def test_inf_handling(memory_tiff_file):
        """
        Test the create_dataset_from_inputs method when the image has input inf values

        """
        imarray = np.array(
            (
                [np.inf, 1, 2, 5],
                [5, 1, 2, 7],
                [-np.inf, 2, 0, 3],
                [4, np.inf, 4, -np.inf],
            )
        )

        with memory_tiff_file(imarray) as left_image_file:
            # Computes the dataset image
            input_config = {
                "left": {
                    "img": left_image_file.name,
                    "nodata": np.inf,
                    "disp": [-60, 0],
                }
            }
            dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # The inf values should be set as -9999
        dst_img_correct = np.array(
            (
                [-9999, 1, 2, 5],
                [5, 1, 2, 7],
                [-9999, 2, 0, 3],
                [4, -9999, 4, -9999],
            )
        )

        np.testing.assert_array_equal(dst_img_correct, dst_left["im"].values)

    @staticmethod
    @pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
    def test_with_full_roi(default_cfg, memory_tiff_file):
        """
        Test the get_window and create_dataset_from_inputs method when the config has a roi

        """
        imarray = np.array(
            (
                [np.inf, 1, 2, 5, 1, 3, 6, 4, 9, 7, 8],
                [5, 1, 2, 7, 1, 4, 7, 8, 5, 8, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [4, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
                [2, 3, 5, 0, 1, 5, 9, 2, 8, 6, 7],
                [1, 2, 4, 5, 2, 6, 7, 7, 3, 7, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [np.inf, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
            )
        )

        # Roi
        roi = {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]}
        roi_gt = imarray[1:8, 1:8]

        # Check create_dataset_from_inputs
        with memory_tiff_file(imarray) as left_image_file:
            input_config = {
                "left": {
                    "img": left_image_file.name,
                    "nodata": default_cfg["input"]["left"]["nodata"],
                    "disp": [-60, 0],
                }
            }
            dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=roi)

        np.testing.assert_array_equal(dst_left["im"].data, roi_gt)

    @staticmethod
    @pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
    def test_with_none_roi(default_cfg, memory_tiff_file):
        """
        Test the create_dataset_from_inputs method when the config hasn't roi

        """
        imarray = np.array(
            (
                [np.inf, 1, 2, 5],
                [5, np.inf, 2, 7],
                [1, 2, np.inf, 3],
            )
        )

        # Check create_dataset_from_inputs
        with memory_tiff_file(imarray) as left_image_file:
            input_config = {
                "left": {
                    "img": left_image_file.name,
                    "nodata": default_cfg["input"]["left"]["nodata"],
                    "disp": [-60, 0],
                }
            }
            dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=None)

        np.testing.assert_array_equal(dst_left["im"].data, imarray)

    @staticmethod
    def test_with_classif_and_roi(default_cfg):
        """
        Test the method create_dataset_from_inputs for the classif and roi

        """
        # Computes the dataset image
        # The classes present in left_classif are "cornfields", "olive tree", "forest"
        input_config = {
            "left": {
                "img": "tests/pandora/left.png",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "classif": "tests/pandora/left_classif.tif",
                "disp": [-60, 0],
            }
        }
        roi = {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]}
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=roi)

        # Classif ground truth
        classif_gt = np.zeros((3, 7, 7))

        # Check the shape
        assert dst_left["im"].shape == (7, 7)
        assert dst_left["classif"].shape == (
            3,
            7,
            7,
        )  # The classes present in left_classif are "cornfields", "olive tree", "forest"

        # Check the data
        np.testing.assert_array_equal(dst_left["classif"].data, classif_gt)

    @staticmethod
    def test_with_segm_and_roi(default_cfg):
        """
        Test the method create_dataset_from_inputs for the segm and roi

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/image/left_img.tif",
                "nodata": default_cfg["input"]["left"]["nodata"],
                "segm": "tests/image/mask_left.tif",
                "disp": [-60, 0],
            }
        }
        roi = {"col": {"first": 1, "last": 3}, "row": {"first": 1, "last": 3}, "margins": [0, 0, 0, 0]}
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=roi)

        # Segmentation ground truth
        segm_gt = np.array(
            [[0, 0, 0], [5, 0, 0], [0, 255, 0]],
        )

        # Check the shape
        assert dst_left["im"].shape == (3, 3)
        assert dst_left["segm"].shape == (3, 3)

        # Check the data
        np.testing.assert_array_equal(dst_left["segm"].data, segm_gt)

    @pytest.mark.parametrize(
        ["input_disparity", "expected"],
        [
            pytest.param(
                "tests/pandora/left_disparity_grid.tif",
                np.array(
                    [
                        rasterio_open("tests/pandora/left_disparity_grid.tif").read(1),
                        rasterio_open("tests/pandora/left_disparity_grid.tif").read(2),
                    ],
                ),
                id="Path to grid file",
            ),
            pytest.param(
                (-60, 0), np.array([np.full((375, 450), -60), np.full((375, 450), 0)]), id="Tuple of integers"
            ),
            pytest.param([-60, 0], np.array([np.full((375, 450), -60), np.full((375, 450), 0)]), id="List of integers"),
        ],
    )
    def test_with_disparity(self, input_disparity, expected):
        """
        Test the method create_dataset_from_inputs with the disparity

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/pandora/left.png",
                "nodata": -9999,
                "disp": input_disparity,
            }
        }

        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        # Check the shape
        assert dst_left["im"].shape == (375, 450)
        assert dst_left["disparity"].shape == (2, 375, 450)

        # Check if the calculated disparity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left["disparity"].data, expected)

    @pytest.mark.parametrize(
        ["img_path", "classif_path", "segm_path"],
        [
            pytest.param(
                "tests/image/left_img.tif",
                "tests/pandora/left_classif.tif",
                None,
                id="between img and classif",
            ),
            pytest.param(
                "tests/pandora/left.png",
                None,
                "tests/image/mask_left.tif",
                id="between img and segm",
            ),
        ],
    )
    def test_fails_with_different_shape(self, img_path, classif_path, segm_path):
        """
        Test with wrong image shapes
        """
        # create dataset
        input_config = {
            "img": img_path,
            "disp": [-60, 0],
            "nodata": -9999,
            "mask": None,
            "classif": classif_path,
            "segm": segm_path,
        }
        with pytest.raises(ValueError):
            create_dataset_from_inputs(input_config=input_config)

    @pytest.fixture()
    def default_image_path(self, memory_tiff_file):
        """
        Create a fake image to test ROI in create_dataset_from_inputs
        """
        imarray = np.array(
            (
                [np.inf, 1, 2, 5, 1, 3, 6, 4, 9, 7, 8],
                [5, 1, 2, 7, 1, 4, 7, 8, 5, 8, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [4, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
                [2, 3, 5, 0, 1, 5, 9, 2, 8, 6, 7],
                [1, 2, 4, 5, 2, 6, 7, 7, 3, 7, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [np.inf, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
            )
        )

        with memory_tiff_file(imarray) as left_image_file:
            yield left_image_file.name

    @pytest.fixture()
    def default_input_roi(self, default_cfg, default_image_path):
        """
        Create an input configuration to test ROI in create_dataset_from_inputs
        """
        input_config = {
            "left": {
                "img": default_image_path,
                "nodata": default_cfg["input"]["left"]["nodata"],
                "disp": [-60, 0],
            }
        }

        return input_config

    @pytest.mark.parametrize(
        ["roi", "expected"],
        [
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                {"row": np.arange(1, 8), "col": np.arange(1, 8)},
                id="ROI inside the image",
            ),
            pytest.param(
                {"col": {"first": 0, "last": 2}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                {"row": np.arange(1, 8), "col": np.arange(0, 5)},
                id="ROI overlap on left side",
            ),
            pytest.param(
                {"col": {"first": 10, "last": 12}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                {"row": np.arange(1, 8), "col": np.arange(8, 11)},
                id="ROI overlap on right side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": -1, "last": 5}, "margins": [2, 2, 2, 2]},
                {"row": np.arange(0, 8), "col": np.arange(1, 8)},
                id="ROI overlap on up side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 9, "last": 11}, "margins": [2, 2, 2, 2]},
                {"row": np.arange(7, 8), "col": np.arange(1, 8)},
                id="ROI overlap on down side",
            ),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
    def test_coords_roi(self, default_input_roi, roi, expected):
        """
        Test the create_dataset_from_inputs method when the config has a roi

        """

        # ROI
        roi_tested = roi

        # Expected coordinates
        coords_gt = expected

        # Input configuration
        input_config = default_input_roi

        # Create dataset with ROI
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"], roi=roi_tested)

        # Test if row coordinates are equals
        np.testing.assert_array_equal(dst_left.coords["row"], coords_gt["row"])

        # Test if col coordinates are equals
        np.testing.assert_array_equal(dst_left.coords["col"], coords_gt["col"])


class TestShiftRightImg:
    """Test shift_right_img function."""

    def test_monoband(self, monoband_image):
        """
        Test shift_right_img_function
        """
        expected = np.array([0.25, 1.25, 2.25, 3.25, 4.25])

        shifted_img = img_tools.shift_right_img(monoband_image, 4)

        # check if col coordinates has been shifted
        np.testing.assert_array_equal(expected, shifted_img[1].col)

    def test_multiband(self, multiband_image):
        """
        Test shift_right_img_function for multiband image
        """
        expected = np.array([0.25, 1.25, 2.25, 3.25, 4.25])

        shifted_img = img_tools.shift_right_img(multiband_image, 4, "r")

        # check if columns coordinates has been shifted
        np.testing.assert_array_equal(expected, shifted_img[1].col)


def test_fuse_classification_bands():
    """
    Test the fuse_classification_bands function
    """
    # Create dataset with input classification map
    # The classes present in left_classif are "cornfields", "olive tree", "forest"
    input_config = {
        "left": {
            "img": "tests/pandora/left.png",
            "nodata": np.nan,
            "classif": "tests/pandora/left_classif.tif",
            "disp": [-60, 0],
        }
    }
    img = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

    # Create ground truth monoband classification map
    gt_monoband_classif = np.zeros((len(img.coords["row"]), len(img.coords["col"])))
    band_index_corn = list(img.band_classif.data).index("cornfields")
    gt_monoband_classif += 1 * img["classif"].data[band_index_corn, :, :]
    band_index_nenuphar = list(img.band_classif.data).index("forest")
    gt_monoband_classif += 2 * img["classif"].data[band_index_nenuphar, :, :]

    # Obtain output monoband classification map
    output_classif_array = img_tools.fuse_classification_bands(img, class_names=["cornfields", "forest"])

    # Check that the obtained classification map is the same as ground truth
    np.testing.assert_array_equal(gt_monoband_classif, output_classif_array)


class TestReadDisp:
    """Test read_disp function."""

    @pytest.mark.parametrize(
        ["input_disparity", "expected"],
        [
            pytest.param(
                "tests/pandora/tiny_left_disparity_grid.tif",
                (np.full((4, 4), -27), np.full((4, 4), -7)),
                id="Path to grid file",
            ),
            pytest.param((-60, 0), (-60, 0), id="Tuple of integers"),
            pytest.param([-60, 0], (-60, 0), id="List of integers"),
        ],
    )
    def test_nominal_case(self, input_disparity, expected):
        """
        Test the funtion read_disp with nominal inputs
        """
        result = img_tools.read_disp(input_disparity)

        # Check if the calculated disparity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(result, expected)

    def test_with_none_as_input(self):
        """
        Test the funtion read_disp with bad input
        """
        with pytest.raises(ValueError, match="disparity should not be None"):
            img_tools.read_disp(None)


class TestGetMetadata:
    """Test get_metadata function."""

    @pytest.fixture()
    def input_cfg(self):
        """Input configuration"""
        return {"input": copy.deepcopy(common.input_cfg_basic)}

    def test_get_metadata_succeed(self, input_cfg):
        """
        Test the method get_metadata with all good information

        """
        # Metadata ground truth
        metadata_gt = xr.Dataset(
            coords={"band_im": [None], "row": np.arange(375), "col": np.arange(450)},
            attrs={"disparity_source": [-60, 0]},
        )

        # get metadata without classif and mask
        metadata_img = img_tools.get_metadata(input_cfg["input"]["left"]["img"], input_cfg["input"]["left"]["disp"])

        # Check that the get_metadata function run whitout error
        assert metadata_img.coords["band_im"] == metadata_gt.coords["band_im"]
        assert (metadata_img.coords["row"] == metadata_gt.coords["row"]).all()
        assert (metadata_img.coords["col"] == metadata_gt.coords["col"]).all()
        assert metadata_img.attrs == metadata_gt.attrs

    @pytest.mark.parametrize(
        ["img_path"],
        [
            pytest.param("tests/pandora/left_fake.png", id="Wrong image path"),
            pytest.param(1, id="Integer for image path"),
            pytest.param(True, id="Boolean for image path"),
        ],
    )
    def test_fail_with_wrong_img_path(self, input_cfg, img_path):
        """
        Test the method get_metadata with wrong information for img param

        """
        with pytest.raises((TypeError, RasterioIOError)):
            img_tools.get_metadata(img=img_path, disparity=input_cfg["input"]["left"]["disp"])

    @pytest.mark.parametrize(
        ["classif", "expected_error"],
        [
            pytest.param(True, "invalid path or file: True", id="Boolean for classification path"),
            pytest.param(1, "invalid path or file: 1", id="Integer for classification path"),
        ],
    )
    def test_fail_with_wrong_classification_param(self, input_cfg, classif, expected_error):
        """
        Test the method get_metadata with wrong information for classif param

        """
        with pytest.raises(TypeError, match=expected_error):
            img_tools.get_metadata(
                img=input_cfg["input"]["left"]["img"], disparity=input_cfg["input"]["left"]["disp"], classif=classif
            )


def test_add_global_disparity(monoband_image):
    """
    Test add_global_disparity function
    """

    dataset = monoband_image

    # add disparity for CARS tiling
    dataset.attrs["disp_min"] = -2
    dataset.attrs["disp_max"] = 2

    test_dataset = img_tools.add_global_disparity(dataset, -2, 2)

    assert test_dataset.attrs["global_disparity"] == [-2, 2]


@pytest.mark.parametrize(
    ["disparities", "expected_error"],
    [
        pytest.param(
            [0, 2],
            "For ambiguity step, the global disparity must be outside the range of the grid disparity",
            id="global_min error",
        ),
        pytest.param(
            [-2, 1],
            "For ambiguity step, the global disparity must be outside the range of the grid disparity",
            id="global_max error",
        ),
        pytest.param(
            [0, 1],
            "For ambiguity step, the global disparity must be outside the range of the grid disparity",
            id="global_extremum error",
        ),
    ],
)
def test_add_global_disparity_failed(monoband_image, disparities, expected_error):
    """
    Test add_global_disparity function
    """

    dataset = monoband_image

    # add disparity for CARS tiling
    dataset.attrs["disp_min"] = -2
    dataset.attrs["disp_max"] = 2

    with pytest.raises(ValueError, match=expected_error):
        _ = img_tools.add_global_disparity(dataset, disparities[0], disparities[1])
