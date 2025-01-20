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

# pylint: skip-file

def cbca(input, cross_left, cross_right, range_col, range_col_right):
    """
    Build the fully aggregated matching cost for one disparity,
    E = S_v(row, col + bottom_arm_length) - S_v(row, col - top_arm_length - 1)

    :param input: cost volume for the current disparity
    :type input: 2D np.array (row, col) dtype = np.float32
    :param cross_left: cross support of the left image
    :type cross_left: 3D np.array (row, col, [left, right, top, bot]) dtype=np.int16
    :param cross_right: cross support of the right image
    :type cross_right: 3D np.array (row, col, [left, right, tpo, bot]) dtype=np.int16
    :param range_col: left column for the current disparity (i.e : np.arrange(nb columns), where the correspondent \
    in the right image is reachable)
    :type range_col: 1D np.array
    :param range_col_right: right column for the current disparity (i.e : np.arrange(nb columns) - disparity, where \
    column - disparity >= 0 and <= nb columns)
    :type range_col_right: 1D np.array
    :return: the fully aggregated matching cost, and the total number of support pixels used for the aggregation
    :rtype: tuple(2D np.array (row , col) dtype = np.float32, 2D np.array (row , col) dtype = np.float32)
    """
    return None, None

def cross_support(image, len_arms, intensity):
    """
    Compute the cross support for an image: find the 4 arms.
    Enforces a minimum support region of 3×3 if pixels are valid.
    The cross support of invalid pixels (pixels that are np.inf) is 0 for the 4 arms.

    :param image: image
    :type image: 2D np.array (row , col) dtype = np.float32
    :param len_arms: maximal length arms
    :param len_arms: int16
    :param intensity: maximal intensity
    :param intensity: float 32
    :return: a 3D np.array ( row, col, [left, right, top, bot] ), with the four arms lengths computes for each pixel
    :rtype:  3D np.array ( row, col, [left, right, top, bot] ), dtype=np.int16
    """
    ...
