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
def compute_matching_costs(img_left, imgs_right, cv, disps, census_width, census_height):
    """
    Given a left image and right images (multiple when doing subpixellic), compute the Census matching costs for all disparities, with the given window.

    :param img_left: the left image
    :type img_left: 2D np.array (row, col) dtype = np.float32
    :param imgs_right: the right images
    :type imgs_right: List of 2D np.array (row, col) dtype = np.float32
    :param cv: cost volume to fill
    :type cv: 3D np.array (row, col, disps) dtype = np.float32
    :param disps: the disparities to sample, sorted
    :type disps: np.array (disps) dtype = np.float32
    :param census_width: the width of the census window
    :type census_width: int
    :param census_height: the height of the census window
    :type census_height: int
    :return: the filled cost volume
    :rtype: 3D np.array (row, col, disps) dtype = np.float32
    """
    ...
