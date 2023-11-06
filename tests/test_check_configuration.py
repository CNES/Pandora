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
This module contains functions to test all the methods in check_configuration module.
"""

import numpy as np
import xarray as xr
from rasterio import Affine
import pytest


from pandora.common import split_inputs
from pandora.img_tools import create_dataset_from_inputs, add_disparity, rasterio_open
from pandora.check_configuration import (
    check_dataset,
    check_datasets,
    default_short_configuration,
    update_conf,
    check_shape,
    check_band_names,
    check_image_dimension,
    check_disparities_from_input,
    check_disparities_from_dataset,
    check_attributes,
)
from tests import common


class TestCheckDataset:
    """Test check_dataset function."""

    @pytest.fixture()
    def dataset_without_bands(self):
        """Build dataset."""
        # Build the default configuration
        input_config = split_inputs(default_short_configuration["input"])
        input_config["left"]["img"] = "tests/image/left_img.tif"
        input_config["left"]["disp"] = [-60, 0]

        # Computes the dataset image
        return create_dataset_from_inputs(input_config=input_config["left"])

    @pytest.fixture()
    def dataset_with_bands(self):
        """Build dataset."""
        # Build the default configuration
        input_config = update_conf(default_short_configuration["input"], common.input_multiband_cfg)  # type: ignore
        input_config_split = split_inputs(input_config)

        # Computes the dataset image
        return create_dataset_from_inputs(input_config=input_config_split["left"])

    @pytest.mark.parametrize(
        ["dataset"],
        [
            pytest.param("dataset_without_bands", id="Without bands"),
            pytest.param("dataset_with_bands", id="With bands"),
        ],
    )
    def test_nominal_case(self, dataset, request):
        """
        Test the nominal case with 2D and 3D image
        """
        check_dataset(request.getfixturevalue(dataset))

    @pytest.mark.parametrize(
        ["array", "dims", "coords"],
        [
            pytest.param(
                np.array(np.full((3, 4), np.nan), dtype=np.float32),
                ["row", "col"],
                {
                    "row": np.arange(3),
                    "col": np.arange(4),
                },
                id="All nan 2D image",
            ),
            pytest.param(
                np.array(np.full((2, 3, 4), np.nan), dtype=np.float32),
                ["band_im", "row", "col"],
                {
                    "band_im": ["r", "g"],
                    "row": np.arange(3),
                    "col": np.arange(4),
                },
                id="All nan 3D image",
            ),
        ],
    )
    def test_fails_with_nan_data(self, array, dims, coords):
        """
        Test image with all nan values
        """
        dataset = xr.Dataset(
            {"im": (dims, array)},
            coords=coords,
        )
        dataset.attrs = {
            "no_data_img": 0,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        dataset.pipe(add_disparity, disparity=[-2, 1], window=None)

        with pytest.raises(SystemExit):
            check_dataset(dataset)


class TestCheckDatasets:
    """Test check_datasets function."""

    def dataset(self, img_path, disparity):
        """Build dataset."""
        input_cfg = {
            "img": img_path,
            "disp": disparity,
            "nodata": -9999,
            "mask": None,
            "classif": None,
            "segm": None,
        }
        return create_dataset_from_inputs(input_cfg)

    @pytest.fixture()
    def datasets(self, request):
        """Build left/right dataset."""
        left_params = request.param[0]
        right_params = request.param[1]
        return self.dataset(*left_params), self.dataset(*right_params)

    @pytest.mark.parametrize(
        "datasets",
        [
            (["tests/pandora/left.png", [-60, 0]], ["tests/pandora/right.png", [0, 60]]),
            (["tests/pandora/left.png", [-60, 0]], ["tests/pandora/right.png", None]),
        ],
        indirect=["datasets"],
    )
    def test_nominal(self, datasets):
        """
        Test the nominal case with image dataset
        """
        dataset_left, dataset_right = datasets
        check_datasets(dataset_left, dataset_right)

    @pytest.mark.parametrize(
        "datasets",
        [(["tests/image/left_img.tif", [-60, 0]], ["tests/pandora/right.png", [0, 60]])],
        indirect=["datasets"],
    )
    def test_fails_with_wrong_dimension(self, datasets):
        """
        Test the nominal case with image dataset
        """
        dataset_left, dataset_right = datasets
        with pytest.raises(SystemExit):
            check_datasets(dataset_left, dataset_right)

    @pytest.mark.parametrize(
        "datasets",
        [
            (["tests/image/left_img.tif", None], ["tests/pandora/right.png", [0, 60]]),
            (["tests/image/left_img.tif", None], ["tests/pandora/right.png", None]),
        ],
        indirect=["datasets"],
    )
    def test_fails_without_disparity(self, datasets):
        """
        Test the nominal case with image dataset
        """
        dataset_left, dataset_right = datasets
        with pytest.raises(SystemExit):
            check_datasets(dataset_left, dataset_right)


class TestCheckBandNames:
    """Test check_band_names function."""

    @pytest.fixture()
    def dataset(self, request):
        """Build dataset."""
        input_cfg = {
            "img": request.param,
            "disp": [-60, 0],
            "nodata": -9999,
            "mask": None,
            "classif": None,
            "segm": None,
        }
        return create_dataset_from_inputs(input_cfg)

    @pytest.mark.parametrize(
        ["img"],
        [
            pytest.param("tests/pandora/left.png", id="image without bands"),
            pytest.param("tests/pandora/left_rgb.tif", id="image with bands"),
        ],
    )
    def test_nominal_case_path(self, img):
        """
        Test the nominal case with image path
        """
        check_band_names(img)

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param("tests/pandora/left.png", id="dataset without bands"),
            pytest.param("tests/pandora/left_rgb.tif", id="dataset with bands"),
        ],
        indirect=["dataset"],
    )
    def test_nominal_case_dataset(self, dataset):
        """
        Test the nominal case with image dataset
        """
        check_band_names(dataset)

    @pytest.mark.parametrize(
        ["bands"],
        [
            pytest.param([0, 1, 2], id="int bands"),
            pytest.param([float(1), float(2), float(3)], id="float bands"),
            pytest.param([np.nan, np.nan, np.nan], id="nan bands"),
        ],
    )
    def test_fails_with_wrong_bands_in_dataset(self, bands):
        """
        Test wrong bands in dataset
        """
        # Create a matching_cost object with simple images
        data = np.array(
            [
                [[7, 8, 1], [4, 5, 2], [8, 9, 10]],
                [[2, 5, 7], [3, 7, 1], [1, 0, 0]],
                [[5, 3, 6], [1, 2, 1], [7, 9, 10]],
            ],
            dtype=np.float64,
        )
        dataset = xr.Dataset(
            {"im": (["band_im", "row", "col"], data)},
            coords={"band_im": bands, "row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        with pytest.raises(SystemExit):
            check_band_names(dataset)


class TestCheckShape:
    """Test check_shape function."""

    @pytest.fixture()
    def dataset(self):
        """Build dataset."""
        input_config = {
            "img": "tests/pandora/left.png",
            "disp": [-60, 0],
            "nodata": -9999,
            "mask": "tests/pandora/occlusion.png",
            "classif": "tests/pandora/left_classif.tif",
            "segm": "tests/pandora/occlusion.png",
        }
        return create_dataset_from_inputs(input_config=input_config)

    @pytest.mark.parametrize(
        ["test"],
        [
            pytest.param("msk", id="with mask"),
            pytest.param("segm", id="with segm"),
            pytest.param("classif", id="with classif"),
        ],
    )
    def test_nominal_case(self, dataset, test):
        """
        Test the nominal case with image, mask, classif and segm wich have the same size
        """
        check_shape(dataset=dataset, ref="im", test=test)

    def test_fails_with_different_shapes(self):
        """
        Test with wrong image shapes
        """
        # create dataset
        input_config = {
            "img": "tests/image/left_img.tif",
            "disp": [-60, 0],
            "nodata": -9999,
            "mask": None,
            "classif": None,
            "segm": None,
        }
        dataset = create_dataset_from_inputs(input_config=input_config)

        data_occlusion = rasterio_open("tests/pandora/occlusion.png").read(1, out_dtype=np.int16, window=None)
        dataset.coords["row_occlusion"] = np.arange(data_occlusion.shape[0])
        dataset.coords["col_occlusion"] = np.arange(data_occlusion.shape[1])
        dataset["occlusion"] = xr.DataArray(data_occlusion, dims=["row_occlusion", "col_occlusion"])

        with pytest.raises(SystemExit):
            check_shape(dataset=dataset, ref="im", test="occlusion")


class TestCheckAttributes:
    """Test check_attribute function."""

    @pytest.fixture()
    def dataset(self):
        """Build dataset."""
        input_config = {
            "img": "tests/image/left_img.tif",
            "disp": [-60, 0],
            "nodata": -9999,
        }

        # Computes the dataset image
        return create_dataset_from_inputs(input_config=input_config)

    @pytest.mark.parametrize("missing_attribute", ["no_data_img", "valid_pixels", "no_data_mask", "crs", "transform"])
    def test_fails_when_given_dataset_with_missing_attribute(self, dataset, missing_attribute):
        """
        Test attributes missing in dataset
        """
        mandatory_attributes = {"no_data_img", "valid_pixels", "no_data_mask", "crs", "transform"}
        del dataset.attrs[missing_attribute]
        with pytest.raises(SystemExit):
            check_attributes(dataset, mandatory_attributes)


class TestCheckImageDimension:
    """Test check_image_dimension function."""

    def test_nominal_case(self):
        """
        Test the nominal case with two image which have same dimension
        """
        img1_ = rasterio_open("tests/pandora/left.png")
        img2_ = rasterio_open("tests/pandora/occlusion.png")
        check_image_dimension(img1_, img2_)

    def test_fails_with_different_image(self):
        """
        Test with two image which have not same dimension
        """
        img1_ = rasterio_open("tests/pandora/left.png")
        img2_ = rasterio_open("tests/image/mask_left.tif")
        with pytest.raises(SystemExit):
            check_image_dimension(img1_, img2_)


class TestCheckDisparitiesFromInput:
    """Test check_disparities_from_input function."""

    @pytest.mark.parametrize(
        ["disparity"],
        [
            pytest.param([-60, 0], id="int list disparities"),
            pytest.param(None, id="None disparity"),
            pytest.param("tests/pandora/left_disparity_grid.tif", id="image path disparities"),
        ],
    )
    def test_nominal_case(self, disparity):
        """
        Test the nominal case with list[int], str, None disparities
        """
        img_left_path_ = "tests/pandora/left.png"
        check_disparities_from_input(disparity, img_left_path_)

    @pytest.mark.parametrize(
        ["disparity", "img_path"],
        [
            pytest.param([60, 0], "tests/pandora/left.png", id="int list disparities"),
            pytest.param(
                "tests/pandora/disp_left.tif", "tests/pandora/left.png", id="image path with one band disparity"
            ),
            pytest.param(
                "tests/pandora/tiny_left_disparity_grid.tif",
                "tests/pandora/left.png",
                id="image disparity with wrong dimension",
            ),
        ],
    )
    def test_fails_with_wrong_disparities(self, disparity, img_path):
        """
        Test with wrong disparities
        """
        with pytest.raises(SystemExit):
            check_disparities_from_input(disparity, img_path)


class TestCheckDisparitiesFromDataset:
    """Test check_disparities_from_dataset function."""

    def test_nominal_case(self):
        """
        Test the nominal case with xarray DataArray disparities
        """
        disparity = xr.DataArray(
            data=np.array(
                [
                    np.full((3, 4), -60),
                    np.full((3, 4), 0),
                ],
                dtype=np.float32,
            ),
            dims=["band_disp", "row", "col"],
            coords={
                "band_disp": ["min", "max"],
                "row": np.arange(3),
                "col": np.arange(4),
            },
        )
        check_disparities_from_dataset(disparity)

    @pytest.mark.parametrize(
        ["disparity"],
        [
            pytest.param(
                xr.DataArray(
                    data=np.array(
                        np.full((3, 4), -60),
                        dtype=np.float32,
                    ),
                    dims=["row", "col"],
                    coords={
                        "row": np.arange(3),
                        "col": np.arange(4),
                    },
                ),
                id="xarray DataArray with one band",
            ),
            pytest.param(
                xr.DataArray(
                    data=np.array(
                        [
                            np.full((3, 4), -60),
                            np.full((3, 4), 0),
                        ],
                        dtype=np.float32,
                    ),
                    dims=["band_disp", "row", "col"],
                    coords={
                        "band_disp": ["max", "min"],
                        "row": np.arange(3),
                        "col": np.arange(4),
                    },
                ),
                id="xarray DataArray disparities with max < min",
            ),
            pytest.param(
                xr.DataArray(
                    data=np.array(
                        [
                            np.full((3, 4), -60),
                            np.full((3, 4), 0),
                        ],
                        dtype=np.float32,
                    ),
                    dims=["band_disp", "row", "col"],
                    coords={
                        "band_disp": ["a", "b"],
                        "row": np.arange(3),
                        "col": np.arange(4),
                    },
                ),
                id="xarray DataArray disparities with wrong band names",
            ),
        ],
    )
    def test_fails_with_wrong_disparities(self, disparity):
        """
        Test with wrong disparities
        """
        with pytest.raises(SystemExit):
            check_disparities_from_dataset(disparity)
