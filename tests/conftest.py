# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
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

"""
Set of fixtures available to all tests.
"""

# pylint: disable=redefined-outer-name

from contextlib import contextmanager
from typing import Union, Iterable, Generator, Callable

import numpy as np
import xarray as xr
import pytest
import rasterio


@pytest.fixture()
def memory_tiff_file() -> Callable:
    """Return an in memory Tiff file as a contextmanager.

    This fixture is a factory fixture in order to be able to use several In memory TIFF files in the same test.
    It returns a contextmanager because once it's closed, the file is lost and can not be reopened to read data.
    """

    @contextmanager
    def context_manager(data: Union[np.ndarray, xr.DataArray]) -> Generator[rasterio.MemoryFile, None, None]:
        """
        Contextmanager that yield a Tiff file stored in memory.

        Data can be of shape (row, col) or (band, row, col). In the latter case, each band of shape
        (row, col) will be stored as a new band in the file.

        Returned object is a `rasterio.MemoryFile` object tha can be used as a file-like object. Data can be read by
        directly passing this object to `rasterio.open` or its `name` attribute which is a string representing a
        virtual file path.

        :param data: data to write into the file.
        :type data: Union[np.ndarray, xr.DataArray]
        :return: MemoryFile with data
        :rtype: Generator[rasterio.MemoryFile, None, None]
        """
        nb_of_bands = 1
        band_indices: Union[int, Iterable[int]] = 1
        if len(data.shape) == 2:
            row, col = data.shape
        else:
            nb_of_bands, row, col = data.shape
            band_indices = range(1, nb_of_bands + 1)
        with rasterio.MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff",
                width=col,
                height=row,
                count=nb_of_bands,
                dtype=data.dtype,
            ) as dataset:
                dataset.write(data, band_indices)
            yield memory_file

    return context_manager


@pytest.fixture()
def left_img_path():
    return "tests/pandora/left.png"


@pytest.fixture()
def right_img_path():
    return "tests/pandora/right.png"


@pytest.fixture()
def correct_input_cfg(disp, left_img_path, right_img_path):
    return {
        "input": {
            "left": {"img": left_img_path, "disp": disp, "nodata": np.nan},
            "right": {
                "img": right_img_path,
                "nodata": np.nan,
            },
        },
    }


@pytest.fixture
def correct_pipeline_cfg(matching_cost_method, window_size, subpix):
    return {
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": matching_cost_method,
                "window_size": window_size,
                "subpix": subpix,
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": "NaN"},
        }
    }
