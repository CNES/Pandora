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
This module contains the required libraries and softwares allowing to execute the software,
and setup elements to configure and identify the software.
"""

from codecs import open as copen
from setuptools import setup, find_packages

CMDCLASS = {}

try:
    from sphinx.setup_command import BuildDoc

    CMDCLASS["build_sphinx"] = BuildDoc
except ImportError:
    print("WARNING: sphinx not available. Doc cannot be built")

REQUIREMENTS = [
    "numpy",
    "xarray>=0.13.*",
    "scipy",
    "rasterio",
    "json-checker",
    "numba>=0.47.*",
    "transitions",
    "scikit-image",
]


REQUIREMENTS_EXTRA = {
    "dev": ["sphinx", "sphinx_rtd_theme", "sphinx_autoapi", "nose2", "pylint", "pre-commit", "mypy"],
    "sgm": ["pandora_plugin_libsgm==0.6.*"],
    "docs": ["sphinx", "sphinx_rtd_theme", "sphinx_autoapi"],
}


def readme():
    with copen("README.md", "r", "utf-8") as fstream:
        return fstream.read()


setup(
    name="pandora",
    version="x.y.z",
    description="Pandora is a stereo matching framework that helps emulate state of the art algorithms",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/CNES/Pandora",
    author="CNES",
    author_email="myriam.cournet@cnes.fr",
    license="Apache License 2.0",
    entry_points={"console_scripts": ["pandora = pandora.Pandora:main"]},
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    extras_require=REQUIREMENTS_EXTRA,
    packages=find_packages(),
    cmdclass=CMDCLASS,
    command_options={
        "build_sphinx": {
            "build_dir": ("setup.py", "doc/build/"),
            "source_dir": ("setup.py", "doc/sources/"),
            "warning_is_error": ("setup.py", True),
        }
    },
)
