#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
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
#
"""
This module contains the required libraries and softwares allowing to execute the software, and setup elements to configure and identify the software. 
"""

from setuptools import setup, find_packages
import subprocess
from codecs import open

cmdclass={}

try:
    from sphinx.setup_command import BuildDoc
    cmdclass['build_sphinx'] = BuildDoc
except ImportError:
    print('WARNING: sphinx not available. Doc cannot be built')

requirements = ['numpy',
                'xarray>=0.13.*',
                'scipy',
                'rasterio',
                'nose2',
                'json-checker',
                'numba>=0.47.*',
                'opencv-python']

REQUIREMENTS_DEV = {'docs': ['sphinx',
                             'sphinx_rtd_theme',
                             'sphinx_autoapi'
                            ]}


def readme():
    with open('README.md', "r", "utf-8") as f:
        return f.read()


setup(name='pandora',
      version='x.y.z',
      description='Pandora is a stereo matching framework that helps emulate state of the art algorithms',
      long_description=readme(),
      url='https://github.com/CNES/Pandora_pandora',
      author='CNES',
      author_email='myriam.cournet@cnes.fr',
      license='Apache License 2.0',
      entry_points={
          'console_scripts': ['pandora = bin.Pandora:main']
      },
      python_requires='>=3.6',
      install_requires=requirements,
      extras_require=REQUIREMENTS_DEV,
      packages=find_packages(),
      cmdclass=cmdclass,
      command_options={
          'build_sphinx': {
              'build_dir': ('setup.py', 'doc/build/'),
              'source_dir': ('setup.py', 'doc/sources/')}},
      )
