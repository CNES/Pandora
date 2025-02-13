#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
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
#
"""
This script provides utilites to automate the getting/setting process
of the current project's version
"""

import os
import subprocess
import sys


def get_vcs():
    """
    Prints the current project's version to stdout
    """
    subprocess.run(["python", "-m", "setuptools_scm"], check=True)


def set_dist(version):
    """
    Sets the dist project version argument
    """
    meson_project_dist_root = os.getenv("MESON_PROJECT_DIST_ROOT")
    meson_rewrite = os.getenv("MESONREWRITE")

    if not meson_project_dist_root or not meson_rewrite:
        print("Error: Required environment variables MESON_PROJECT_DIST_ROOT or MESONREWRITE are missing.")
        sys.exit(1)

    print(meson_project_dist_root)
    rewrite_command = f"{meson_rewrite} --sourcedir {meson_project_dist_root} "
    rewrite_command += f"kwargs set project / version {version}"

    subprocess.run(rewrite_command.split(" "), check=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <get-vcs | set-dist> [version]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "get-vcs":
        get_vcs()
    elif command == "set-dist":
        if len(sys.argv) < 3:
            print("Error: Missing version argument for set-dist.")
            sys.exit(1)
        set_dist(sys.argv[2])
    else:
        print("Error: Invalid command.")
        sys.exit(1)
