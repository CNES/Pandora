# pylint:disable=too-many-arguments
# pylint:disable=too-many-lines
# pylint: disable=too-many-public-methods
# !/usr/bin/env python
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
This module contains class associated to the pandora state machine
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Union, List

import numpy as np
import xarray as xr

try:
    import graphviz  # pylint: disable=unused-import
    from transitions.extensions import GraphMachine as Machine

    FLAG_GRAPHVIZ = True
except ImportError:
    from transitions import Machine  # type: ignore

    FLAG_GRAPHVIZ = False
from transitions import MachineError

from pandora import (  # pylint: disable=redefined-builtin
    aggregation,
    disparity,
    filter,
    multiscale,
    optimization,
    refinement,
    matching_cost,
    semantic_segmentation,
)
from pandora.margins import GlobalMargins

from pandora.criteria import validity_mask

from pandora import validation
from pandora import cost_volume_confidence
from .img_tools import prepare_pyramid


# This module contains class associated to the pandora state machine


class PandoraMachine(Machine):  # pylint:disable=too-many-instance-attributes
    """
    PandoraMachine class to create and use a state machine
    """

    _transitions_run = [
        {
            "trigger": "matching_cost",
            "source": "begin",
            "dest": "cost_volume",
            "prepare": "matching_cost_prepare",
            "before": "matching_cost_run",
        },
        {
            "trigger": "aggregation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "aggregation_run",
        },
        {
            "trigger": "semantic_segmentation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "semantic_segmentation_run",
        },
        {
            "trigger": "optimization",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "optimization_run",
        },
        {
            "trigger": "disparity",
            "source": "cost_volume",
            "dest": "disp_map",
            "before": "disparity_run",
        },
        {
            "trigger": "filter",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "filter_run",
        },
        {
            "trigger": "refinement",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "refinement_run",
        },
        {
            "trigger": "validation",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "validation_run",
        },
        # Conditional state change, if it is the last scale the multiscale state will not be triggered
        # This way, after the last scale we can still apply a filter state
        {
            "trigger": "multiscale",
            "source": "disp_map",
            "conditions": "is_not_last_scale",
            "dest": "begin",
            "before": "run_multiscale",
        },
        {
            "trigger": "cost_volume_confidence",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "cost_volume_confidence_run",
        },
    ]

    _transitions_check = [
        {
            "trigger": "check_matching_cost",
            "source": "begin",
            "dest": "cost_volume",
            "before": "matching_cost_check_conf",
        },
        {
            "trigger": "check_aggregation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "aggregation_check_conf",
        },
        {
            "trigger": "check_semantic_segmentation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "semantic_segmentation_check_conf",
        },
        {
            "trigger": "check_optimization",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "optimization_check_conf",
        },
        {
            "trigger": "check_disparity",
            "source": "cost_volume",
            "dest": "disp_map",
            "before": "disparity_check_conf",
        },
        {
            "trigger": "check_filter",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "filter_check_conf",
        },
        {
            "trigger": "check_refinement",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "refinement_check_conf",
        },
        {
            "trigger": "check_validation",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "validation_check_conf",
        },
        # For the check conf we define the destination of multiscale state as disp_map instead of begin
        # given the conditional change of state
        {
            "trigger": "check_multiscale",
            "source": "disp_map",
            "dest": "disp_map",
            "before": "multiscale_check_conf",
        },
        {
            "trigger": "check_cost_volume_confidence",
            "source": "cost_volume",
            "dest": "cost_volume",
            "before": "cost_volume_confidence_check_conf",
        },
    ]

    def __init__(self) -> None:
        """
        Initialize Pandora Machine

        :return: None
        """
        # Left image scale pyramid
        self.img_left_pyramid: List[xr.Dataset] = [None]
        # Right image scale pyramid
        self.img_right_pyramid: List[xr.Dataset] = [None]
        # Left image
        self.left_img: xr.Dataset = None
        # Right image
        self.right_img: xr.Dataset = None
        # Minimum disparity
        self.disp_min: np.ndarray = None
        # Maximum disparity
        self.disp_max: np.ndarray = None
        # Maximum disparity for the right image
        self.right_disp_min: np.ndarray = None
        # Minimum disparity for the right image
        self.right_disp_max: np.ndarray = None
        # User minimum disparity
        self.dmin_user: np.ndarray = None
        # User maximum disparity
        self.dmax_user: np.ndarray = None
        # User minimum disparity right
        self.dmin_user_right: np.ndarray = None
        # User maximum disparity right
        self.dmax_user_right: np.ndarray = None

        # Scale factor
        self.scale_factor: int = None
        # Number of scales
        self.num_scales: int = 1
        # Current scale
        self.current_scale: int = None

        # left cost volume
        self.left_cv: xr.Dataset = None
        # right cost volume
        self.right_cv: xr.Dataset = None
        # left disparity map
        self.left_disparity: xr.Dataset = None
        # right disparity map
        self.right_disparity: xr.Dataset = None

        self.step: int = 1

        # Pandora's pipeline configuration
        self.pipeline_cfg: Dict = {"pipeline": {}}

        # Margins that cumulates:
        self.margins = GlobalMargins()
        # Right disparity map computation information: Can be None or "cross_checking_accurate"
        # Useful during the running steps to choose if we must compute left and right objects.
        self.right_disp_map = None
        # Define avalaible states
        states_ = ["begin", "cost_volume", "disp_map"]

        # Instance matching_cost
        self.matching_cost_: Union[matching_cost.AbstractMatchingCost, None] = None

        if FLAG_GRAPHVIZ:
            # Initialize a machine without any transition
            Machine.__init__(
                self,
                states=states_,
                initial="begin",
                transitions=None,
                auto_transitions=False,
                use_pygraphviz=False,
            )
        else:
            # Initialize a machine without any transition
            Machine.__init__(
                self,
                states=states_,
                initial="begin",
                transitions=None,
                auto_transitions=False,
            )

        logging.getLogger("transitions").setLevel(logging.WARNING)

    def matching_cost_prepare(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Matching cost computation
        :param cfg: user configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        self.matching_cost_ = matching_cost.AbstractMatchingCost(**cfg["pipeline"][input_step])  # type: ignore
        # Update min and max disparity according to the current scale
        self.disp_min = self.disp_min * self.scale_factor
        self.disp_max = self.disp_max * self.scale_factor
        self.left_cv = self.matching_cost_.allocate_cost_volume(self.left_img, (self.disp_min, self.disp_max), cfg)

        # Compute validity mask to identify invalid points in cost volume
        self.left_cv = validity_mask(self.left_img, self.right_img, self.left_cv)

        if self.right_disp_map == "cross_checking_accurate":
            # Update min and max disparity according to the current scale
            self.right_disp_min = self.right_disp_min * self.scale_factor
            self.right_disp_max = self.right_disp_max * self.scale_factor
            self.right_cv = self.matching_cost_.allocate_cost_volume(
                self.right_img, (self.right_disp_min, self.right_disp_max), cfg
            )

            # Compute validity mask to identify invalid points in cost volume
            self.right_cv = validity_mask(self.right_img, self.left_img, self.right_cv)

    def matching_cost_run(self, _: Dict[str, dict], __: str) -> None:
        """
        Matching cost computation
        :return: None
        """
        logging.info("Matching cost computation...")

        # Compute cost volume and mask it
        self.left_cv = self.matching_cost_.compute_cost_volume(self.left_img, self.right_img, self.left_cv)

        # Conversion to np.nan of masked points in left cost_volume
        self.matching_cost_.cv_masked(
            self.left_img,
            self.right_img,
            self.left_cv,
            self.disp_min,
            self.disp_max,
        )

        if self.right_disp_map == "cross_checking_accurate":
            # Compute right cost volume and mask it
            self.right_cv = self.matching_cost_.compute_cost_volume(self.right_img, self.left_img, self.right_cv)

            # Conversion to np.nan of masked points in right cost_volume
            self.matching_cost_.cv_masked(
                self.right_img,
                self.left_img,
                self.right_cv,
                self.right_disp_min,
                self.right_disp_max,
            )

    def aggregation_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Cost (support) aggregation
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Aggregation computation...")
        aggregation_ = aggregation.AbstractAggregation(**cfg["pipeline"][input_step])  # type: ignore
        aggregation_.cost_volume_aggregation(self.left_img, self.right_img, self.left_cv)
        if self.right_disp_map == "cross_checking_accurate":
            aggregation_.cost_volume_aggregation(self.right_img, self.left_img, self.right_cv)

    def semantic_segmentation_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Building semantic segmentation computation
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Semantic segmentation computation...")
        semantic_segmentation_ = semantic_segmentation.AbstractSemanticSegmentation(
            self.left_img, **cfg["pipeline"][input_step]
        )  # type: ignore
        self.left_img = semantic_segmentation_.compute_semantic_segmentation(
            self.left_cv, self.left_img, self.right_img
        )
        if self.right_disp_map == "cross_checking_accurate":
            self.right_img = semantic_segmentation_.compute_semantic_segmentation(
                self.right_cv, self.right_img, self.left_img
            )

    def optimization_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Cost optimization
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Cost optimization...")
        optimization_ = optimization.AbstractOptimization(self.left_img, **cfg["pipeline"][input_step])  # type: ignore

        self.left_cv = optimization_.optimize_cv(self.left_cv, self.left_img, self.right_img)
        if self.right_disp_map == "cross_checking_accurate":
            self.right_cv = optimization_.optimize_cv(self.right_cv, self.right_img, self.left_img)

    def disparity_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Disparity computation and validity mask
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Disparity computation...")
        disparity_ = disparity.AbstractDisparity(**cfg["pipeline"][input_step])  # type: ignore

        self.left_disparity = disparity_.to_disp(self.left_cv, self.left_img, self.right_img)

        if self.right_disp_map == "cross_checking_accurate":
            self.right_disparity = disparity_.to_disp(self.right_cv, self.right_img, self.left_img)

    def filter_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Disparity filter
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Disparity filtering...")
        filter_ = filter.AbstractFilter(
            cfg=cfg["pipeline"][input_step],
            image_shape=(self.left_img.sizes["row"], self.left_img.sizes["col"]),
            step=self.step,
        )  # type: ignore
        filter_.filter_disparity(self.left_disparity)
        if self.right_disp_map == "cross_checking_accurate":
            filter_.filter_disparity(self.right_disparity)

    def refinement_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Subpixel disparity refinement
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Subpixel refinement...")
        refinement_ = refinement.AbstractRefinement(**cfg["pipeline"][input_step])  # type: ignore

        refinement_.subpixel_refinement(self.left_cv, self.left_disparity)
        if self.right_disp_map == "cross_checking_accurate":
            refinement_.subpixel_refinement(self.right_cv, self.right_disparity)

    def validation_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Validation of disparity map
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Validation...")
        validation_ = validation.AbstractValidation(**cfg["pipeline"][input_step])  # type: ignore

        self.left_disparity = validation_.disparity_checking(self.left_disparity, self.right_disparity)
        if self.right_disp_map == "cross_checking_accurate":
            self.right_disparity = validation_.disparity_checking(self.right_disparity, self.left_disparity)
            # Interpolated mismatch and occlusions
            if "interpolated_disparity" in cfg["pipeline"][input_step]:
                interpolate_ = validation.AbstractInterpolation(**cfg["pipeline"][input_step])  # type: ignore
                interpolate_.interpolated_disparity(self.left_disparity)
                interpolate_.interpolated_disparity(self.right_disparity)

    def run_multiscale(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Compute the disparity range for the next scale
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Disparity range computation...")
        multiscale_ = multiscale.AbstractMultiscale(
            self.left_img, self.right_img, **cfg["pipeline"][input_step]
        )  # type: ignore

        # Update min and max user disparity according to the current scale
        self.dmin_user = self.dmin_user * self.scale_factor
        self.dmax_user = self.dmax_user * self.scale_factor

        # Compute disparity range for the next scale level
        self.disp_min, self.disp_max = multiscale_.disparity_range(self.left_disparity, self.dmin_user, self.dmax_user)
        # Set to None the disparity map for the next scale
        self.left_disparity = None

        if self.right_disp_map == "cross_checking_accurate":
            # Update min and max user disparity according to the current scale
            self.dmin_user_right = self.dmin_user_right * self.scale_factor
            self.dmax_user_right = self.dmax_user_right * self.scale_factor
            self.right_disp_min, self.right_disp_max = multiscale_.disparity_range(
                self.right_disparity, self.dmin_user_right, self.dmax_user_right
            )
            self.right_disparity = None

        # Get the next scale's images
        self.left_img = self.img_left_pyramid.pop(0)
        self.right_img = self.img_right_pyramid.pop(0)

        # Update the current scale for the next state
        self.current_scale = self.current_scale - 1

    def cost_volume_confidence_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Confidence prediction
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        logging.info("Cost volume confidence computation...")
        # In case multiple confidence maps are computed, add its
        # name to the indicator to distinguish the different maps
        cfg["pipeline"][input_step]["indicator"] = ""
        if len(input_step.split(".")) == 2:
            cfg["pipeline"][input_step]["indicator"] = "." + input_step.split(".")[1]
        confidence_ = cost_volume_confidence.AbstractCostVolumeConfidence(**cfg["pipeline"][input_step])  # type: ignore

        logging.info("Confidence prediction...")

        self.left_disparity, self.left_cv = confidence_.confidence_prediction(
            self.left_disparity, self.left_img, self.right_img, self.left_cv
        )
        if self.right_disp_map == "cross_checking_accurate":
            self.right_disparity, self.right_cv = confidence_.confidence_prediction(
                self.right_disparity, self.right_img, self.left_img, self.right_cv
            )

    def run_prepare(
        self,
        cfg: Dict[str, dict],
        left_img: xr.Dataset,
        right_img: xr.Dataset,
        scale_factor: Union[None, int] = None,
        num_scales: Union[None, int] = None,
    ) -> None:
        """
        Prepare the machine before running

        :param cfg:  configuration
        :type cfg: dict
        :param left_img: left Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type left_img: xarray.Dataset
        :param right_img: right Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type right_img: xarray.Dataset
        :param scale_factor: scale factor for multiscale
        :type scale_factor: int or None
        :param num_scales: scales number for multiscale
        :type num_scales: int or None
        :return: None
        """

        # Mono-resolution processing by default if num_scales or scale_factor are not specified
        if num_scales is None or scale_factor is None:
            self.num_scales = 1
            self.scale_factor = 1
        else:
            self.num_scales = num_scales
            self.scale_factor = scale_factor

        if self.num_scales > 1:
            # If multiscale processing, create pyramid and select first scale's images
            self.img_left_pyramid, self.img_right_pyramid = prepare_pyramid(
                left_img, right_img, self.num_scales, scale_factor
            )
            self.left_img = self.img_left_pyramid.pop(0)
            self.right_img = self.img_right_pyramid.pop(0)
            # Initialize current scale
            self.current_scale = num_scales - 1
            # If multiscale, disparities can only be int.
            # Downscale disparities since the pyramid is processed from coarse to original size
            self.disp_min = left_img["disparity"].sel(band_disp="min") / (self.scale_factor**self.num_scales)
            self.disp_max = left_img["disparity"].sel(band_disp="max") / (self.scale_factor**self.num_scales)
            # User disparity
            self.dmin_user = self.disp_min
            self.dmax_user = self.disp_max
            # If multiscale disparities can only be int, and right disparity can only be np.ndarray, so right disparity
            # can not be defined in the input conf
            # Right disparities
            self.right_disp_min = -self.disp_max
            self.right_disp_max = -self.disp_min
            # Right user disparity
            self.dmin_user_right = self.right_disp_min
            self.dmax_user_right = self.right_disp_max
        else:
            # If no multiscale processing, select the original images
            self.left_img = left_img
            self.right_img = right_img
            # If no multiscale processing, current scale is zero
            self.current_scale = 0
            # Disparities
            self.disp_min = left_img["disparity"].sel(band_disp="min").data
            self.disp_max = left_img["disparity"].sel(band_disp="max").data
            # Right disparities
            if "disparity" in right_img.data_vars:
                self.right_disp_min = right_img["disparity"].sel(band_disp="min").data
                self.right_disp_max = right_img["disparity"].sel(band_disp="max").data
            else:
                self.right_disp_min = -left_img["disparity"].sel(band_disp="max").data
                self.right_disp_max = -left_img["disparity"].sel(band_disp="min").data

        # Initiate output disparity datasets
        self.left_disparity = xr.Dataset()
        self.right_disparity = xr.Dataset()
        # To determine whether the right disparity map has to be computed
        if "validation" in cfg["pipeline"]:
            self.right_disp_map = cfg["pipeline"]["validation"]["validation_method"]
        # Add transitions
        self.add_transitions(self._transitions_run)

    def run(self, input_step: str, cfg: Dict[str, dict]) -> None:
        """
        Run pandora step by triggering the corresponding machine transition

        :param input_step: step to trigger
        :type input_step: str
        :param cfg: pipeline configuration
        :type  cfg: dict
        :return: None
        """

        try:
            # There may be steps that are repeated several times, for example:
            #     'filter': {
            #       'filter_method': 'median'
            #     },
            #     'filter.1': {
            #       'filter_method': 'bilateral'
            #     }
            # But there's only a filter transition. Therefore, in the case of filter.1 we have to call the
            # filter
            # trigger and give the configuration of filter.1
            step_to_trigger = input_step.split(".")[0]
            self.trigger(step_to_trigger, cfg, input_step)
        except (MachineError, KeyError, AttributeError):
            logging.error("A problem occurs during Pandora running %s  step. Be sure of your sequencement", input_step)
            raise

    def run_exit(self) -> None:
        """
        Clear transitions and return to state begin

        :return: None
        """
        self.remove_transitions(self._transitions_run)
        self.set_state("begin")

    def matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the matching cost configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        # Create matching_cost object to check its step configuration
        matching_cost_ = matching_cost.AbstractMatchingCost(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = matching_cost_.cfg
        self.step = matching_cost_.cfg["step"]
        self.margins.add_cumulative(input_step, matching_cost_.margins)

        # Check the coherence between the band selected for the matching_cost step
        # and the bands present on left and right image
        self.check_band_pipeline(
            self.left_img.coords["band_im"].data,
            cfg["matching_cost"]["matching_cost_method"],
            matching_cost_.cfg["band"],
        )
        self.check_band_pipeline(
            self.right_img.coords["band_im"].data,
            cfg["matching_cost"]["matching_cost_method"],
            matching_cost_.cfg["band"],
        )

    def disparity_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        disparity_ = disparity.AbstractDisparity(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = disparity_.cfg
        self.margins.add_cumulative(input_step, disparity_.margins)

    def filter_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the filter configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        filter_config = copy.deepcopy(cfg[input_step])
        filter_ = filter.AbstractFilter(
            cfg=filter_config,
            image_shape=(self.left_img.sizes["row"], self.left_img.sizes["col"]),
            step=self.step,
        )  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = filter_.cfg
        self.margins.add_non_cumulative(input_step, filter_.margins)

    def refinement_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the refinement configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        refinement_ = refinement.AbstractRefinement(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = refinement_.cfg
        self.margins.add_cumulative(input_step, refinement_.margins)

    def aggregation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the aggregation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        aggregation_ = aggregation.AbstractAggregation(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = aggregation_.cfg
        self.margins.add_cumulative(input_step, aggregation_.margins)

    def semantic_segmentation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the semantic_segmentation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        semantic_segmentation_ = semantic_segmentation.AbstractSemanticSegmentation(
            self.left_img, **cfg[input_step]
        )  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = semantic_segmentation_.cfg

        # If semantic_segmentation is present, check that the necessary bands are present in the inputs
        self.check_band_pipeline(
            self.left_img.coords["band_im"].data,
            cfg["semantic_segmentation"]["segmentation_method"],
            cfg["semantic_segmentation"]["RGB_bands"],
        )
        # If vegetation_band is present in semantic_segmentation, check that the bands are present
        # in the input left classification
        if "vegetation_band" in cfg["semantic_segmentation"]:
            if "classif" not in self.left_img.data_vars:
                raise ValueError(
                    "For performing the semantic_segmentation step in the pipeline, "
                    "classif must be present in left image."
                )
            self.check_band_pipeline(
                self.left_img.coords["band_classif"].data,
                cfg["semantic_segmentation"]["segmentation_method"],
                cfg["semantic_segmentation"]["vegetation_band"]["classes"],
            )

    def optimization_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the optimization configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        # When SGM optimization is used the only permitted step value is 1
        if self.step != 1:
            raise AttributeError("For performing the SGM optimization step, step attribute must be equal to 1")

        optimization_ = optimization.AbstractOptimization(self.left_img, **cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = optimization_.cfg

        # If geometric_prior is needed for the optimization step,
        # check that the necessary inputs and bands are present
        if "geometric_prior" in cfg["optimization"]:
            source = cfg["optimization"]["geometric_prior"]["source"]
            if source in ["classif", "segm"]:
                if source not in self.left_img.data_vars:
                    raise AttributeError(
                        f"For performing the 3SGM optimization step in the pipeline left {source} must be present."
                    )
                # If sgm optimization is present with geometric_prior classification, check that the
                # classes bands are present in the input classification
                if "classes" in cfg["optimization"]["geometric_prior"]:
                    self.check_band_pipeline(
                        self.left_img.coords["band_classif"].data,
                        cfg["optimization"]["optimization_method"],
                        cfg["optimization"]["geometric_prior"]["classes"],
                    )
        self.margins.add_cumulative(input_step, optimization_.margins)

    def validation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the validation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        validation_ = validation.AbstractValidation(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = validation_.cfg
        if "interpolated_disparity" in validation_.cfg:
            _ = validation.AbstractInterpolation(  # type:ignore
                **cfg[input_step]
            )

        self.right_disp_map = validation_.cfg["validation_method"]

        # If left disparities are grids of disparity and the right disparities are none, the cross-checking
        # method cannot be used
        if (
            isinstance(self.left_img.attrs["disparity_source"], str)
            and self.right_img.attrs["disparity_source"] is None
        ):
            raise AttributeError(
                "The cross-checking step cannot be processed if disp_min, disp_max are paths to the "
                "left disparity grids and disp_right_min, disp_right_max are none."
            )

    def multiscale_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: disparity computation configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        multiscale_ = multiscale.AbstractMultiscale(self.left_img, self.right_img, **cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = multiscale_.cfg

    def cost_volume_confidence_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the confidence configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        confidence_ = cost_volume_confidence.AbstractCostVolumeConfidence(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = confidence_.cfg

    def check_conf(
        self, cfg: Dict[str, dict], img_left: xr.Dataset, img_right: xr.Dataset, right_left_img_check: bool = False
    ) -> None:
        """
        Check configuration and transitions

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param img_left: image left with metadata
        :type  img_left: xarray.Dataset
        :param img_right: image right with metadata
        :type  img_right: xarray.Dataset
        :param right_left_img_check: if right image has been checked
        :type right_left_img_check: bool
        :return: None
        """

        self.left_img = img_left
        self.right_img = img_right

        # Add transitions to the empty machine.
        self.add_transitions(self._transitions_check)

        for input_step in list(cfg["pipeline"]):
            try:
                # There may be steps that are repeated several times, for example:
                #     'filter': {
                #       'filter_method': 'median'
                #     },
                #     'filter.1': {
                #       'filter_method': 'bilateral'
                #     }
                # But there's only a filter transition. Therefore, in the case of filter.1 we have to call the
                # filter
                # trigger and give the configuration of filter.1

                # change input name to avoid may_[step] repetition in transitions packages
                check_input = "check_" + input_step

                if len(input_step.split(".")) != 1:
                    self.trigger(check_input.split(".")[0], cfg["pipeline"], input_step)
                else:
                    self.trigger(check_input, cfg["pipeline"], input_step)

            except (MachineError, KeyError, AttributeError):
                raise MachineError("A problem occurs during Pandora checking. Be sure of your sequencing")

        # Remove transitions
        self.remove_transitions(self._transitions_check)

        # Coming back to the initial state
        self.set_state("begin")

        # second round RIGHT/LEFT
        if self.right_disp_map and not right_left_img_check:
            self.check_conf(cfg, img_right, img_left, True)
            self.left_img = img_left
            self.right_img = img_right

    def remove_transitions(self, transition_list: List[Dict[str, str]]) -> None:
        """
        Delete all transitions defined in the input list

        :param transition_list: list of transitions
        :type transition_list: dict
        :return: None
        """
        # Transition is removed using trigger name. But one trigger name can be used by multiple transitions
        # In this case, the 'remove_transition' function removes all transitions using this trigger name
        # deleted_triggers list is used to avoid multiple call of 'remove_transition' with the same trigger name.
        deleted_triggers = []
        for trans in transition_list:
            if trans not in deleted_triggers:
                self.remove_transition(trans["trigger"])
                deleted_triggers.append(trans["trigger"])

    def is_not_last_scale(self, _: str, __: Dict[str, dict]) -> bool:
        """
        Check if the current scale is the last scale
        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: boolean
        """

        if self.current_scale == 0:
            return False
        return True

    @staticmethod
    def check_band_pipeline(band_list: np.ndarray, step: str, band_used: Union[None, str, List[str], Dict]) -> None:
        """
        Check coherence band parameter between pipeline step and image dataset

        :param band_list: band names of image
        :type band_list: numpy.ndarray with bands
        :param step: pipeline step
        :type step: str
        :param band_used: band names for pipeline step
        :type band_used: None, str, List[str] or Dict
        :return: None
        """

        # If no bands are given, then the input image shall be monoband
        if not band_used:
            if len(band_list) != 1:
                raise AttributeError(f"Missing band instantiate on {step} step : input image is multiband")
        # check that the image have the band names
        elif isinstance(band_used, dict):
            for _, band in band_used.items():
                if band not in band_list:
                    raise AttributeError(f"Wrong band instantiate on {step} step: {band} not in input image")
        else:
            for band in band_used:
                if band not in band_list:
                    raise AttributeError(f"Wrong band instantiate on {step} step: {band} not in input image")
