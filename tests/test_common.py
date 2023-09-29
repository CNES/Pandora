"""Test common module."""
import pytest

from pandora.common import split_inputs


class TestSplitInputs:
    """Tests functions to split inputs in left and right keys."""

    # TOBEREMOVED: Remove this function after ticket #314
    @pytest.fixture()
    def inputs_param(self):
        return {
            "img_left": "path_to_left_img",
            "img_right": "img_right_path",
            "nodata_left": 666,
            "nodata_right": 999,
            "disp_left": 23,
            "disp_right": 32,
            "left_mask": "path_to_left_mask",
            "right_mask": "path_to_right_mask",
            "left_classif": "left_classif_path",
            "right_classif": "path_to_right_classif",
            "left_segm": "left_segm_path",
            "right_segm": "right_segm_path",
        }

    def test(self, inputs_param):
        inputs = split_inputs(inputs_param)
        assert inputs == {
            "left": {
                "img": "path_to_left_img",
                "nodata": 666,
                "disp": 23,
                "mask": "path_to_left_mask",
                "classif": "left_classif_path",
                "segm": "left_segm_path",
            },
            "right": {
                "img": "img_right_path",
                "nodata": 999,
                "disp": 32,
                "mask": "path_to_right_mask",
                "classif": "path_to_right_classif",
                "segm": "right_segm_path",
            },
        }
