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
This module contains functions to test all the methods in img_tools module.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import rasterio
import xarray as xr
from skimage.io import imsave

import pandora
from pandora.common import split_inputs
from pandora import img_tools
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
    return common.matching_cost_tests_multiband_setup()[0]  # type: ignore


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


class TestBuildDatasetFromInputs:
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
                "nodata": default_cfg["input"]["nodata_left"],
                "mask": "tests/image/mask_left.tif",
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
                "nodata": default_cfg["input"]["nodata_left"],
                "classif": "tests/pandora/left_classif.tif",
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
                "nodata": default_cfg["input"]["nodata_left"],
                "classif": "tests/pandora/left_classif.tif",
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
    def test_with_segm(default_cfg):
        """
        Test the method create_dataset_from_inputs for the segmentation

        """
        # Computes the dataset image
        input_config = {
            "left": {
                "img": "tests/image/left_img.tif",
                "nodata": default_cfg["input"]["nodata_left"],
                "segm": "tests/image/mask_left.tif",
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
        input_config = {"left": {"img": "tests/pandora/left.png", "nodata": default_cfg["input"]["nodata_left"]}}
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
        input_config = {"left": {"img": "tests/image/left_img.tif", "nodata": default_cfg["input"]["nodata_left"]}}
        dst_left = img_tools.create_dataset_from_inputs(input_config=input_config["left"])

        gt_crs = None
        gt_transform = None

        # Check if the CRS and Transform are correctly set to None
        np.testing.assert_array_equal(gt_crs, dst_left.attrs["crs"])
        np.testing.assert_array_equal(gt_transform, dst_left.attrs["transform"])

    @staticmethod
    def test_inf_handling(tmp_path):
        """
        Test the create_dataset_from_inputs method when the image has input inf values

        """
        image_path = tmp_path / "left_img.tif"
        imarray = np.array(
            (
                [np.inf, 1, 2, 5],
                [5, 1, 2, 7],
                [-np.inf, 2, 0, 3],
                [4, np.inf, 4, -np.inf],
            )
        )
        imsave(str(image_path), imarray, plugin="tifffile", photometric="MINISBLACK")

        # Computes the dataset image
        input_config = {"left": {"img": str(image_path), "nodata": np.inf}}
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


class TestCheckDataset:
    """Test check_dataset function."""

    @pytest.fixture()
    def dataset(self):
        """Build dataset."""
        # Build the default configuration
        default_cfg = pandora.check_configuration.default_short_configuration
        input_config = split_inputs(default_cfg["input"])
        input_config["left"]["img"] = "tests/image/left_img.tif"

        # Computes the dataset image
        return img_tools.create_dataset_from_inputs(input_config=input_config["left"])

    def test_nominal_case(self, dataset):
        """Should not raise error."""
        img_tools.check_dataset(dataset)

    @pytest.mark.parametrize("missing_attribute", ["no_data_img", "valid_pixels", "no_data_mask", "crs", "transform"])
    def test_failing_when_given_dateset_with_missing_attribute(self, dataset, missing_attribute):
        del dataset.attrs[missing_attribute]
        with pytest.raises(SystemExit):
            img_tools.check_dataset(dataset)


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
