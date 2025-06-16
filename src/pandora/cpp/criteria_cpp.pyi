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
import numpy as np
from typing import Tuple

def partially_missing_variable_ranges(disps: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Returns a mask of the pixels with a partially missing variable range in the right image.

    :param disps: Disparity range, np.ndarray of shape (2, width, height)
    :type disps: np.ndarray(float)
    :param img_mask: Mask of valid pixels in the right image, np.ndarray of shape (width, height)
    :type img_mask: np.ndarray(bool)
    """
    ...
