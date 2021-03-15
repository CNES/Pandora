# pylint: disable=invalid-name
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
This module contains the general function to run Pandora pipeline.
"""

import argparse

import pandora


def get_parser():
    """
    ArgumentParser for Pandora

    :return parser
    """
    parser = argparse.ArgumentParser(description="Pandora stereo matching")
    parser.add_argument(
        "config",
        help="Path to a json file containing the input files paths and algorithm parameters",
    )
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser


def main():
    """
    Call Pandora's main
    """
    # Get parser
    parser = get_parser()
    args = parser.parse_args()

    # Run the Pandora pipeline
    pandora.main(args.config, args.output_dir, args.verbose)


if __name__ == "__main__":
    main()
