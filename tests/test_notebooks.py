# pylint:disable=line-too-long
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the Pandora notebooks.
"""
import subprocess
import tempfile
import unittest
import fileinput
import pytest


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora notebooks
    """

    @staticmethod
    @pytest.mark.notebook_with_sgm
    @pytest.mark.notebook_tests
    def test_statistical_and_visual_analysis():
        """
        Test that the statistical_and_visual_analysis notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            # Convert notebook to script
            subprocess.run(
                [
                    f"jupyter nbconvert --to script notebooks/statistical_and_visual_analysis.ipynb --output-dir {directory}"
                ],
                shell=True,
                check=False,
            )
            # Copy data into the temporary directory
            subprocess.run(
                ["cp -r " "notebooks/data " f"{directory}"],
                shell=True,
                check=True,
            )
            # Copy notebook snippets/utils.py
            subprocess.run(
                ["cp -r " "notebooks/snippets " f"{directory}"],
                shell=True,
                check=True,
            )
            # Deactivate matplotlib and bokeh show from utils.py file
            for line in fileinput.input(
                f"{directory}/snippets/utils.py",
                inplace=True,
            ):
                # Deactivate bokeh show of fig and layout
                if "show(fig)" in line:
                    line = line.replace("show(fig)", "file_html(fig, CDN)")
                if "show(layout)" in line:
                    line = line.replace("show(layout)", "file_html(layout, CDN)")
                # Deactivate matplotlib show of fig
                if "fig.show()" in line:
                    line = line.replace("fig.show()", 'fig.write_html("test.html")')
                # Deactivate JupyterDash app
                if "app = JupyterDash" in line:
                    line = "app = None"
                print(line)  # This print must be present, otherwise the file is empty
            # run notebook
            out = subprocess.run(
                [f"ipython {directory}/statistical_and_visual_analysis.py"],
                shell=True,
                check=False,
                cwd=directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        assert out.returncode == 0

    @staticmethod
    @pytest.mark.notebook_tests
    @pytest.mark.notebook_pandora
    def test_usage_with_multiscale():
        """
        Test that the usage_with_multiscale notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [f"jupyter nbconvert --to script notebooks/usage_with_multiscale.ipynb --output-dir {directory}"],
                shell=True,
                check=False,
            )

            out = subprocess.run(
                [f"ipython {directory}/usage_with_multiscale.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    @staticmethod
    @pytest.mark.notebook_tests
    @pytest.mark.notebook_pandora
    def test_introduction_and_basic_usage():
        """
        Test that the introduction_and_basic_usage notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    f"jupyter nbconvert --to script notebooks/introduction_and_basic_usage.ipynb --output-dir {directory}"
                ],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/introduction_and_basic_usage.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    @staticmethod
    @pytest.mark.notebook_tests
    def test_api_check_conf():
        """
        Test that the api_check_conf notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    f"jupyter nbconvert --to script notebooks/advanced_examples/api_check_conf.ipynb --output-dir {directory}"
                ],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/api_check_conf.py"],
                shell=True,
                check=False,
                cwd="notebooks/advanced_examples",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0
