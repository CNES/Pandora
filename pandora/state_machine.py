# pylint:disable=too-many-arguments
# !/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
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

import warnings
import logging
from typing import Dict, Union, List
import numpy as np
import xarray as xr
from json_checker import Checker, And

try:
    import graphviz  # pylint: disable=unused-import
    from transitions.extensions import GraphMachine as Machine
except ImportError:
    from transitions import Machine
from transitions import MachineError

from pandora import aggregation
from pandora import disparity
from pandora import filter  # pylint: disable=redefined-builtin
from pandora import multiscale
from pandora import optimization
from pandora import refinement
from pandora import matching_cost

# This silences numba's TBB threading layer warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
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
            "after": "matching_cost_run",
        },
        {
            "trigger": "aggregation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "aggregation_run",
        },
        {
            "trigger": "optimization",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "optimization_run",
        },
        {
            "trigger": "disparity",
            "source": "cost_volume",
            "dest": "disp_map",
            "after": "disparity_run",
        },
        {
            "trigger": "filter",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "filter_run",
        },
        {
            "trigger": "refinement",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "refinement_run",
        },
        {
            "trigger": "validation",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "validation_run",
        },
        # Conditional state change, if it is the last scale the multiscale state will not be triggered
        # This way, after the last scale we can still apply a filter state
        {
            "trigger": "multiscale",
            "source": "disp_map",
            "conditions": "is_not_last_scale",
            "dest": "begin",
            "after": "run_multiscale",
        },
        {
            "trigger": "cost_volume_confidence",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "cost_volume_confidence_run",
        },
    ]

    _transitions_check = [
        {
            "trigger": "matching_cost",
            "source": "begin",
            "dest": "cost_volume",
            "before": "right_disp_map_check_conf",
            "after": "matching_cost_check_conf",
        },
        {
            "trigger": "aggregation",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "aggregation_check_conf",
        },
        {
            "trigger": "optimization",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "optimization_check_conf",
        },
        {
            "trigger": "disparity",
            "source": "cost_volume",
            "dest": "disp_map",
            "after": "disparity_check_conf",
        },
        {
            "trigger": "filter",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "filter_check_conf",
        },
        {
            "trigger": "refinement",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "refinement_check_conf",
        },
        {
            "trigger": "validation",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "validation_check_conf",
        },
        # For the check conf we define the destination of multiscale state as disp_map instead of begin
        # given the conditional change of state
        {
            "trigger": "multiscale",
            "source": "disp_map",
            "dest": "disp_map",
            "after": "multiscale_check_conf",
        },
        {
            "trigger": "cost_volume_confidence",
            "source": "cost_volume",
            "dest": "cost_volume",
            "after": "cost_volume_confidence_check_conf",
        },
    ]

    def __init__(
        self,
        img_left_pyramid: List[xr.Dataset] = None,
        img_right_pyramid: List[xr.Dataset] = None,
        disp_min: Union[np.array, int] = None,
        disp_max: Union[np.array, int] = None,
        right_disp_min: Union[np.array, None] = None,
        right_disp_max: Union[np.array, None] = None,
    ) -> None:

        """
        Initialize Pandora Machine

        :param img_left_pyramid: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left_pyramid: xarray.Dataset
        :param img_right_pyramid: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right_pyramid: xarray.Dataset
        :param disp_min: minimal disparity
        :type disp_min: int or np.array
        :param disp_max: maximal disparity
        :type disp_max: int or np.array
        :param right_disp_min: minimal disparity of the right image
        :type right_disp_min: None or np.array
        :param right_disp_max: maximal disparity of the right image
        :type right_disp_max: None or np.array
        :return: None
        """
        # Left image scale pyramid
        self.img_left_pyramid: List[xr.Dataset] = img_left_pyramid
        # Right image scale pyramid
        self.img_right_pyramid: List[xr.Dataset] = img_right_pyramid
        # Left image
        self.left_img: xr.Dataset = None
        # Right image
        self.right_img: xr.Dataset = None
        # Minimum disparity
        self.disp_min: Union[np.array, int] = disp_min
        # Maximum disparity
        self.disp_max: Union[np.array, int] = disp_max
        # Maximum disparity for the right image
        self.right_disp_min: Union[np.array, None] = right_disp_min
        # Minimum disparity for the right image
        self.right_disp_max: Union[np.array, None] = right_disp_max
        # User minimum disparity
        self.dmin_user: Union[np.array, int] = None
        # User maximum disparity
        self.dmax_user: Union[np.array, int] = None
        # User minimum disparity right
        self.dmin_user_right: Union[np.array, None] = None
        # User maximum disparity right
        self.dmax_user_right: Union[np.array, None] = None

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

        # Pandora's pipeline configuration
        self.pipeline_cfg: Dict = {"pipeline": {}}
        # Right disparity map computation information: Can be "none" or "accurate"
        # Useful during the running steps to choose if we must compute left and right objects.
        self.right_disp_map = "none"
        # Define avalaible states
        states_ = ["begin", "cost_volume", "disp_map"]

        # Initialize a machine without any transition
        Machine.__init__(
            self,
            states=states_,
            initial="begin",
            transitions=None,
            auto_transitions=False,
        )

        logging.getLogger("transitions").setLevel(logging.WARNING)

    def matching_cost_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Matching cost computation
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """

        logging.info("Matching cost computation...")
        matching_cost_ = matching_cost.AbstractMatchingCost(**cfg[input_step])  # type: ignore

        # Update min and max disparity according to the current scale
        self.disp_min = self.disp_min * self.scale_factor
        self.disp_max = self.disp_max * self.scale_factor
        # Obtain absolute min and max disparities
        dmin_min, dmax_max = matching_cost_.dmin_dmax(self.disp_min, self.disp_max)

        # Compute cost volume and mask it
        if self.right_disp_map != "accurate":
            self.left_cv = matching_cost_.compute_cost_volume(self.left_img, self.right_img, dmin_min, dmax_max)
            matching_cost_.cv_masked(
                self.left_img,
                self.right_img,
                self.left_cv,
                self.disp_min,
                self.disp_max,
            )

        else:
            self.left_cv = matching_cost_.compute_cost_volume(self.left_img, self.right_img, dmin_min, dmax_max)
            matching_cost_.cv_masked(
                self.left_img,
                self.right_img,
                self.left_cv,
                self.disp_min,
                self.disp_max,
            )

            # Update min and max disparity according to the current scale
            self.right_disp_min = self.right_disp_min * self.scale_factor
            self.right_disp_max = self.right_disp_max * self.scale_factor
            # Obtain absolute min and max right disparities
            dmin_min_right, dmax_max_right = matching_cost_.dmin_dmax(self.right_disp_min, self.right_disp_max)
            # Compute right cost volume and mask it
            self.right_cv = matching_cost_.compute_cost_volume(
                self.right_img, self.left_img, dmin_min_right, dmax_max_right
            )
            matching_cost_.cv_masked(
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
        aggregation_ = aggregation.AbstractAggregation(**cfg[input_step])  # type: ignore
        if self.right_disp_map != "accurate":
            aggregation_.cost_volume_aggregation(self.left_img, self.right_img, self.left_cv)
        else:
            aggregation_.cost_volume_aggregation(self.left_img, self.right_img, self.left_cv)
            aggregation_.cost_volume_aggregation(self.right_img, self.left_img, self.right_cv)

    def optimization_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Cost optimization
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        optimization_ = optimization.AbstractOptimization(**cfg[input_step])  # type: ignore
        logging.info("Cost optimization...")
        if self.right_disp_map != "accurate":
            self.left_cv = optimization_.optimize_cv(self.left_cv, self.left_img, self.right_img)
        else:
            self.left_cv = optimization_.optimize_cv(self.left_cv, self.left_img, self.right_img)
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
        disparity_ = disparity.AbstractDisparity(**cfg[input_step])  # type: ignore
        if self.right_disp_map != "accurate":
            self.left_disparity = disparity_.to_disp(self.left_cv, self.left_img, self.right_img)
            disparity_.validity_mask(self.left_disparity, self.left_img, self.right_img, self.left_cv)
        elif self.right_disp_map == "accurate":
            self.left_disparity = disparity_.to_disp(self.left_cv, self.left_img, self.right_img)
            disparity_.validity_mask(self.left_disparity, self.left_img, self.right_img, self.left_cv)
            self.right_disparity = disparity_.to_disp(self.right_cv, self.right_img, self.left_img)
            disparity_.validity_mask(self.right_disparity, self.right_img, self.left_img, self.right_cv)

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
        filter_ = filter.AbstractFilter(**cfg[input_step])  # type: ignore
        if self.right_disp_map != "accurate":
            filter_.filter_disparity(self.left_disparity)
        else:
            filter_.filter_disparity(self.left_disparity)
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
        refinement_ = refinement.AbstractRefinement(**cfg[input_step])  # type: ignore
        logging.info("Subpixel refinement...")
        if self.right_disp_map != "accurate":
            refinement_.subpixel_refinement(self.left_cv, self.left_disparity)
        else:
            refinement_.subpixel_refinement(self.left_cv, self.left_disparity)
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
        validation_ = validation.AbstractValidation(**cfg[input_step])  # type: ignore

        logging.info("Validation...")

        if self.right_disp_map != "accurate":
            self.left_disparity = validation_.disparity_checking(self.left_disparity, self.right_disparity)

        else:
            self.left_disparity = validation_.disparity_checking(self.left_disparity, self.right_disparity)
            self.right_disparity = validation_.disparity_checking(self.right_disparity, self.left_disparity)
            # Interpolated mismatch and occlusions
            if "interpolated_disparity" in cfg[input_step]:
                interpolate_ = validation.AbstractInterpolation(**cfg[input_step])  # type: ignore
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

        multiscale_ = multiscale.AbstractMultiscale(**cfg[input_step])  # type: ignore

        # Update min and max user disparity according to the current scale
        self.dmin_user = self.dmin_user * self.scale_factor
        self.dmax_user = self.dmax_user * self.scale_factor

        # Compute disparity range for the next scale level
        if self.right_disp_map != "accurate":
            self.disp_min, self.disp_max = multiscale_.disparity_range(
                self.left_disparity, self.dmin_user, self.dmax_user
            )
            # Set to None the disparity map for the next scale
            self.left_disparity = None

        else:
            self.disp_min, self.disp_max = multiscale_.disparity_range(
                self.left_disparity, self.dmin_user, self.dmax_user
            )
            # Set to None the disparity map for the next scale
            self.left_disparity = None

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
        confidence_ = cost_volume_confidence.AbstractCostVolumeConfidence(**cfg[input_step])  # type: ignore

        logging.info("Confidence prediction...")

        if self.right_disp_map == "none":
            self.left_disparity, self.left_cv = confidence_.confidence_prediction(
                self.left_disparity, self.left_img, self.right_img, self.left_cv
            )
        else:
            self.left_disparity, self.left_cv = confidence_.confidence_prediction(
                self.left_disparity, self.left_img, self.right_img, self.left_cv
            )
            self.right_disparity, self.right_cv = confidence_.confidence_prediction(
                self.right_disparity, self.right_img, self.left_img, self.right_cv
            )

    def run_prepare(
        self,
        cfg: Dict[str, dict],
        left_img: xr.Dataset,
        right_img: xr.Dataset,
        disp_min: Union[np.array, int],
        disp_max: Union[np.array, int],
        scale_factor: Union[None, int] = None,
        num_scales: Union[None, int] = None,
        right_disp_min: Union[None, np.array] = None,
        right_disp_max: Union[None, np.array] = None,
    ) -> None:
        """
        Prepare the machine before running

        :param cfg:  configuration
        :type cfg: dict
        :param left_img: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type left_img: xarray.Dataset
        :param right_img: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type right_img: xarray.Dataset
        :param disp_min: minimal disparity
        :type disp_min: int or np.array
        :param disp_max: maximal disparity
        :type disp_max: int or np.array
        :param scale_factor: scale factor for multiscale
        :type scale_factor: int or None
        :param num_scales: scales number for multiscale
        :type num_scales: int or None
        :param disp_min_right: minimal disparity of the right image
        :type disp_min_right: np.array or None
        :param disp_max_right: maximal disparity of the right image
        :type disp_max_right: np.array or None
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
            self.disp_min = int(disp_min / (scale_factor ** self.num_scales))
            self.disp_max = int(disp_max / (scale_factor ** self.num_scales))
            # User disparity
            self.dmin_user = self.disp_min
            self.dmax_user = self.disp_max
            # If multiscale disparities can only be int, and right disparity can only be np.array, so right disparity
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
            self.disp_min = disp_min
            self.disp_max = disp_max
            # Right disparities
            if right_disp_max is not None and right_disp_min is not None:
                # If the right disparity was defined in the input conf
                self.right_disp_min = right_disp_min
                self.right_disp_max = right_disp_max
            else:
                self.right_disp_min = -self.disp_max
                self.right_disp_max = -self.disp_min

        # Initiate output disparity datasets
        self.left_disparity = xr.Dataset()
        self.right_disparity = xr.Dataset()
        # To determine whether the right disparity map has to be computed
        self.right_disp_map = cfg["right_disp_map"]["method"]
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
            if len(input_step.split(".")) != 1:
                self.trigger(input_step.split(".")[0], cfg, input_step)
            else:
                self.trigger(input_step, cfg, input_step)
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

    def right_disp_map_check_conf(
        self, cfg: Dict[str, dict], input_step: str  # pylint:disable=unused-argument
    ) -> None:
        """
        Check the right_disp_map configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        schema = {"method": And(str, lambda input: "none" or "accurate")}

        checker = Checker(schema)
        checker.validate(cfg["right_disp_map"])

        # Store the righ disparity map configuration
        self.right_disp_map = cfg["right_disp_map"]["method"]

    def matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the matching cost configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        matching_cost_ = matching_cost.AbstractMatchingCost(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = matching_cost_.cfg

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

    def filter_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the filter configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        filter_ = filter.AbstractFilter(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = filter_.cfg

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

    def optimization_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the optimization configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """
        optimization_ = optimization.AbstractOptimization(**cfg[input_step])  # type: ignore
        self.pipeline_cfg["pipeline"][input_step] = optimization_.cfg

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
            interpolate_ = validation.AbstractInterpolation(  # type:ignore # pylint:disable=unused-variable
                **cfg[input_step]
            )

        if validation_.cfg["validation_method"] == "cross_checking" and self.right_disp_map != "accurate":
            raise MachineError(
                "Can t trigger event cross-checking validation  if right_disp_map method equal to "
                + self.right_disp_map
            )

    def multiscale_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: disparity computation configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        multiscale_ = multiscale.AbstractMultiscale(**cfg[input_step])  # type: ignore
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

    def check_conf(self, cfg: Dict[str, dict]) -> None:
        """
        Check configuration and transitions

        :param cfg: pipeline configuration
        :type  cfg: dict
        :return:
        """

        # Add transitions to the empty machine.
        self.add_transitions(self._transitions_check)

        # Warning: first element of cfg dictionary is not a transition. It contains information about the way to
        # compute right disparity map.
        for input_step in list(cfg)[1:]:
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

                if len(input_step.split(".")) != 1:
                    self.trigger(input_step.split(".")[0], cfg, input_step)
                else:
                    self.trigger(input_step, cfg, input_step)

            except (MachineError, KeyError, AttributeError):
                logging.error("A problem occurs during Pandora checking. Be sure of your sequencement")
                raise

        # Remove transitions
        self.remove_transitions(self._transitions_check)

        # Coming back to the initial state
        self.set_state("begin")

    def remove_transitions(self, transition_list: List[Dict[str, str]]) -> None:
        """
        Delete all transitions defined in the input list

        :param transition_list: list of transitions
        :type transition_list: dict
        :return: None
        """
        # Transition is removed using trigger name. But one trigger name can be used by multiple transitions
        # In this case, the 'remove_transition' function removes all transitions using this trigger name
        # deleted_triggers list is used to avoid multiple call of 'remove_transition'' with the same trigger name.
        deleted_triggers = []
        for trans in transition_list:
            if trans not in deleted_triggers:
                self.remove_transition(trans["trigger"])
                deleted_triggers.append(trans["trigger"])

    def is_not_last_scale(self, input_step: str, cfg: Dict[str, dict]) -> bool:  # pylint:disable=unused-argument
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
