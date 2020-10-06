from transitions import MachineError
from transitions import Machine
import logging
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Union
from json_checker import Checker, Or, And

from pandora import stereo
from pandora import disparity
from pandora import filter
from pandora import refinement
from pandora import aggregation
from pandora import optimization
from pandora import validation
from pandora import common

"""
This module contains class associated to the pandora state machine
"""


class PandoraMachine(Machine):

    """
    PandoraMachine class to create and use a state machine
    """

    _transitions_run = [
        {'trigger': 'stereo', 'source': 'begin', 'dest': 'cost_volume','after': 'stereo_run'},
        {'trigger': 'aggregation', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'aggregation_run'},
        {'trigger': 'optimization', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'optimization_run'},
        {'trigger': 'disparity', 'source': 'cost_volume', 'dest': 'disp_map', 'after': 'disparity_run'},
        {'trigger': 'filter', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'filter_run'},
        {'trigger': 'refinement', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'refinement_run'},
        {'trigger': 'validation', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'validation_run'},
        {'trigger': 'resize', 'source': 'disp_map', 'dest': 'resized_disparity', 'after': 'resize_run'}
    ]

    _transitions_check = [
        {'trigger': 'stereo', 'source': 'begin', 'dest': 'cost_volume','before': 'right_disp_map_check_conf',
         'after': 'stereo_check_conf'},
        {'trigger': 'aggregation', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'aggregation_check_conf'},
        {'trigger': 'optimization', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'optimization_check_conf'},
        {'trigger': 'disparity', 'source': 'cost_volume', 'dest': 'disp_map', 'after': 'disparity_check_conf'},
        {'trigger': 'filter', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'filter_check_conf'},
        {'trigger': 'refinement', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'refinement_check_conf'},
        {'trigger': 'validation', 'source': 'disp_map', 'dest': 'disp_map', 'after': 'validation_check_conf'},
        {'trigger': 'resize', 'source': 'disp_map', 'dest': 'resized_disparity', 'after': 'resize_check_conf'}
    ]

    def __init__(self, left_img: xr.Dataset =None, right_img: xr.Dataset =None,
                 disp_min: Union[int, np.ndarray] =None, disp_max: Union[int, np.ndarray] =None,
                 right_disp_min: Union[int, np.ndarray] =None, right_disp_max: Union[int, np.ndarray] =None):
        """
        Initialize Pandora Machine

        :param left_img: left Dataset image
        :type left_img:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param right_img: right Dataset image
        :type right_img:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param disp_min: minimal disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximal disparity
        :type disp_max: int or np.ndarray
        :param right_disp_min: minimal disparity of the right image
        :type right_disp_min: None, int or np.ndarray
        :param right_disp_max: maximal disparity of the right image
        :type right_disp_max: None, int or np.ndarray
        """

        self.left_img = left_img
        self.right_img = right_img
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.right_disp_min = right_disp_min
        self.right_disp_max = right_disp_max

        # left cost volume
        self.left_cv = None
        # right cost volume
        self.right_cv = None
        # left disparity map
        self.left_disparity = None
        # right disparity map
        self.right_disparity = None

        # Pandora's pipeline configuration
        self.pipeline_cfg = {'pipeline': {}}
        # Right disparity map computation information: Can be "none" or "accurate"
        # Usefull during the running steps to choose if we must compute left and right objects.
        self.right_disp_map = "none"
        # Define avalaible states
        states_ = ['begin', 'cost_volume', 'disp_map', 'resized_disparity']

        # Initiliaze a machine without any transition
        Machine.__init__(self, states=states_, initial='begin', transitions=None, auto_transitions=False)

        logging.getLogger("transitions").setLevel(logging.WARNING)


    def stereo_run(self, cfg: Dict[str, dict], input_step: str):
        """
        Matching cost computation
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """

        logging.info('Matching cost computation...')
        stereo_ = stereo.AbstractStereo(**cfg['pipeline'][input_step])
        dmin_min, dmax_max = stereo_.dmin_dmax(self.disp_min, self.disp_max)

        if not self.right_disp_map == "accurate":
            self.left_cv = stereo_.compute_cost_volume(self.left_img, self.right_img, dmin_min, dmax_max)
            self.left_cv = stereo_.cv_masked(self.left_img, self.right_img, self.left_cv, self.disp_min, self.disp_max,
)

        else:
            self.left_cv = stereo_.compute_cost_volume(self.left_img, self.right_img, dmin_min, dmax_max)
            self.left_cv = stereo_.cv_masked(self.left_img, self.right_img, self.left_cv, self.disp_min, self.disp_max)

            if self.right_disp_min is None:
                self.right_disp_min = -self.disp_max
                self.right_disp_max = -self.disp_min

            dmin_min_sec, dmax_max_sec = stereo_.dmin_dmax(self.right_disp_min, self.right_disp_max)
            self.right_cv = stereo_.compute_cost_volume(self.right_img, self.left_img, dmin_min_sec, dmax_max_sec,
)
            self.right_cv = stereo_.cv_masked(self.right_img, self.left_img, self.right_cv, self.right_disp_min,
                                              self.right_disp_max)


    def aggregation_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Cost (support) aggregation
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        aggregation_ = aggregation.AbstractAggregation(**cfg['pipeline'][input_step])
        if not self.right_disp_map == "accurate":
            self.left_cv = aggregation_.cost_volume_aggregation(self.left_img, self.right_img, self.left_cv)
        else:
            self.left_cv = aggregation_.cost_volume_aggregation(self.left_img, self.right_img, self.left_cv)
            self.right_cv = aggregation_.cost_volume_aggregation(self.right_img, self.left_img, self.right_cv)

    def optimization_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Cost optimization
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        optimization_ = optimization.AbstractOptimization(**cfg['pipeline'][input_step])
        logging.info('Cost optimization...')
        if not self.right_disp_map == "accurate":
            self.left_cv = optimization_.optimize_cv(self.left_cv, self.left_img, self.right_img)
        else:
            self.left_cv = optimization_.optimize_cv(self.left_cv, self.left_img, self.right_img)
            self.right_cv = optimization_.optimize_cv(self.right_cv, self.right_img, self.left_img)

    def disparity_run(self, cfg: Dict[str, dict], input_step: str):
        """
        Disparity computation and validity mask
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        logging.info('Disparity computation...')
        disparity_ = disparity.AbstractDisparity(**cfg['pipeline'][input_step])
        if self.right_disp_map == "none":
            self.left_disparity = disparity_.to_disp(self.left_cv, self.left_img, self.right_img)
            self.left_disparity = disparity_.validity_mask(self.left_disparity, self.left_img, self.right_img,
                                                           self.left_cv)
        elif self.right_disp_map == "accurate":
            self.left_disparity = disparity_.to_disp(self.left_cv, self.left_img, self.right_img)
            self.left_disparity = disparity_.validity_mask(self.left_disparity, self.left_img, self.right_img,
                                                           self.left_cv)
            self.right_disparity = disparity_.to_disp(self.right_cv, self.right_img, self.left_img)
            self.right_disparity = disparity_.validity_mask(self.right_disparity, self.right_img, self.left_img,
                                                         self.right_cv)


    def filter_run(self, cfg: Dict[str, dict], input_step: str):
        """
        Disparity filter
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        logging.info('Disparity filtering...')
        filter_ = filter.AbstractFilter(**cfg['pipeline'][input_step])
        if self.right_disp_map == "none":
            self.left_disparity = filter_.filter_disparity(self.left_disparity)
        else:
            self.left_disparity = filter_.filter_disparity(self.left_disparity)
            self.right_disparity = filter_.filter_disparity(self.right_disparity)

    def refinement_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Subpixel disparity refinement
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        refinement_ = refinement.AbstractRefinement(**cfg['pipeline'][input_step])
        logging.info('Subpixel refinement...')
        if self.right_disp_map == "none":
            self.left_cv, self.left_disparity = refinement_.subpixel_refinement(self.left_cv, self.left_disparity)
        else:
            self.left_cv, self.left_disparity = refinement_.subpixel_refinement(self.left_cv, self.left_disparity)
            self.right_cv, self.right_disparity = refinement_.subpixel_refinement(self.right_cv, self.right_disparity)


    def validation_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Validation of disparity map
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        validation_ = validation.AbstractValidation(**cfg['pipeline'][input_step])

        logging.info('Validation...')

        if self.right_disp_map == "none":
            self.left_disparity = validation_.disparity_checking(self.left_disparity, self.right_disparity)

        else:
            self.left_disparity = validation_.disparity_checking(self.left_disparity, self.right_disparity)
            self.right_disparity = validation_.disparity_checking(self.right_disparity, self.left_disparity)

            # Interpolated mismatch and occlusions
            if 'interpolated_disparity' in cfg:
                interpolate_ = validation.AbstractInterpolation(**cfg['pipeline'][input_step])
                self.left_disparity = interpolate_.interpolated_disparity(self.left_disparity)
                self.right_disparity = interpolate_.interpolated_disparity(self.right_disparity)

    def resize_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Resize left disparity map
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        logging.info('Resize disparity map...')
        if self.right_disp_map == "none":
            self.left_disparity = common.resize(self.left_disparity, cfg['pipeline'][input_step]['border_disparity'])

        else:
            self.left_disparity = common.resize(self.left_disparity, cfg['pipeline'][input_step]['border_disparity'])
            self.right_disparity = common.resize(self.right_disparity, cfg['pipeline'][input_step]['border_disparity'])


    def run_prepare(self,cfg: Dict[str, dict], left_img: xr.Dataset, right_img: xr.Dataset, disp_min: Union[int, np.ndarray],
                    disp_max: Union[int, np.ndarray], right_disp_min: Union[None, int, np.ndarray] = None,
                    right_disp_max: Union[None, int, np.ndarray] = None):
        """
        Prepare the machine before running
        :param cfg:  configuration
        :type  cfg: dict
        :param left_img: left Dataset image
        :type left_img:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param right_img: right Dataset image
        :type right_img:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param disp_min: minimal disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximal disparity
        :type disp_max: int or np.ndarray
        :param right_disp_min: minimal disparity of the right image
        :type right_disp_min: None, int or np.ndarray
        :param right_disp_max: maximal disparity of the right image
        :type right_disp_max: None, int or np.ndarray
        """

        self.left_img = left_img
        self.right_img = right_img
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.right_disp_max = right_disp_max
        self.right_disp_min = right_disp_min
        self.left_disparity = xr.Dataset()
        self.right_disparity = xr.Dataset()

        self.right_disp_map = cfg['pipeline']['right_disp_map']['method']

        self.add_transitions(self._transitions_run)

    def run(self, input_step: str, cfg: Dict[str, dict]):
        """
        Run pandora step by triggering the corresponding machine transition

        :param input_step: step to trigger
        :type input_step: str
        :param cfg: pipeline configuration
        :type  cfg: dict
        :return:
        """
        try:
            # There may be steps that are repeated several times, for example:
            #     "filter": {
            #       "filter_method": "median"
            #     },
            #     "filter_1": {
            #       "filter_method": "bilateral"
            #     }
            # But there's only a filter transition. Therefore, in the case of filter_1 we have to call the
            # filter
            # trigger and give the configuration of filter_1
            if len(input_step.split('_')) != 1:
                self.trigger(input_step.split('_')[0], cfg, input_step)
            else:
                self.trigger(input_step, cfg, input_step)
        except (MachineError, KeyError):
            print('\n A problem occurs during Pandora running ' + input_step +
                  '. Be sure of your sequencement step  \n')
            raise

    def run_exit(self):

        self.remove_transitions(self._transitions_run)
        self.set_state('begin')


    def right_disp_map_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the right_disp_map configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        schema = {
            "method": And(str, lambda x: 'none' or 'accurate')
        }

        checker = Checker(schema)
        checker.validate(cfg['right_disp_map'])

        # Store the righ disparity map configuration
        self.right_disp_map = cfg['right_disp_map']['method']


    def stereo_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the stereo configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        stereo_ = stereo.AbstractStereo(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = stereo_.cfg

    def disparity_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        disparity_ = disparity.AbstractDisparity(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = disparity_.cfg

    def filter_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the filter configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        filter_ = filter.AbstractFilter(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = filter_.cfg

    def refinement_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the refinement configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        refinement_ = refinement.AbstractRefinement(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = refinement_.cfg

    def aggregation_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the aggregation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        aggregation_ = aggregation.AbstractAggregation(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = aggregation_.cfg

    def optimization_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the optimization configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        optimization_ = optimization.AbstractOptimization(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = optimization_.cfg

    def validation_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the validation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        validation_ = validation.AbstractValidation(**cfg[input_step])
        self.pipeline_cfg['pipeline'][input_step] = validation_.cfg
        if 'interpolated_disparity' in validation_.cfg:
            interpolate_ = validation.AbstractInterpolation(**cfg[input_step])

        if validation_.cfg['validation_method'] == "cross_checking" and self.right_disp_map == "none":
            raise MachineError("Can't trigger event cross-checking validation  if right_disp_map method equal to "
                               + self.right_disp_map )


    def resize_check_conf(self, cfg: Dict[str, dict], input_step: str):
        """
        Check the resize configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        schema = {
            "border_disparity": Or(int, float, lambda x: np.isnan(x)),
        }

        checker = Checker(schema)
        checker.validate(cfg[input_step])

    def check_conf(self, cfg: Dict[str, dict]):
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
                #     "filter": {
                #       "filter_method": "median"
                #     },
                #     "filter_1": {
                #       "filter_method": "bilateral"
                #     }
                # But there's only a filter transition. Therefore, in the case of filter_1 we have to call the
                # filter
                # trigger and give the configuration of filter_1

                if len(input_step.split('_')) != 1:
                    self.trigger(input_step.split('_')[0], cfg, input_step)
                else:
                    self.trigger(input_step, cfg, input_step)

            except (MachineError, KeyError):
                print('\n Problem during Pandora checking configuration steps sequencing. '
                    'Check your configuration file. \n')
                raise

        # Remove transitions
        self.remove_transitions(self._transitions_check)

        # Coming back to the initial state
        self.set_state('begin')

    def remove_transitions(self, transition_list: Dict[str, dict]):
        """
        Delete all transitions defined in the input list

        :param transition_list: list of transitions
        :type transition_list: dict
        :return:
        """
        # Transition is removed using trigger name. But one trigger name can be used by multiple transitions
        # In this case, the "remove_transition" function removes all transitions using this trigger name
        # deleted_triggers list is used to avoid multiple call of "remove_transition" with the same trigger name.
        deleted_triggers = []
        for trans in transition_list:
            if trans['trigger'] not in deleted_triggers:
                self.remove_transition(trans['trigger'])
                deleted_triggers.append(trans['trigger'])
