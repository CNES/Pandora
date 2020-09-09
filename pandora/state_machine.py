from transitions import MachineError
from transitions import Machine
import logging
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Union

from pandora import stereo
from pandora import disparity
from pandora import filter
from pandora import refinement
from pandora import aggregation
from pandora import optimization
from pandora import validation

"""
This module contains class associated to the pandora state machine
"""


class PandoraMachine(Machine):

    """
    PandoraMachine class to create and use a state machine
    """

    _transitions_run = [
        {'trigger': 'stereo', 'source': 'begin', 'dest': 'cost_volume', 'after': 'stereo_run'},
        {'trigger': 'aggregation', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'aggregation_run'},
        {'trigger': 'optimization', 'source': 'cost_volume', 'dest': 'cost_volume', 'after': 'optimization_run'},
        {'trigger': 'disparity', 'source': 'cost_volume', 'dest': 'reference_disparity', 'after': 'disparity_run'},
        {'trigger': 'filter', 'source': 'reference_disparity', 'dest': 'reference_disparity',
         'after': 'filter_run'},
        {'trigger': 'refinement', 'source': 'reference_disparity', 'dest': 'reference_disparity',
         'after': 'refinement_run'},
        {'trigger': 'validation', 'source': 'reference_disparity', 'dest': 'reference_and_secondary_disparity',
         'after': 'validation_run'},
        {'trigger': 'filter', 'source': 'reference_and_secondary_disparity',
         'dest': 'reference_and_secondary_disparity', 'after': 'filter_run_ref_and_sec'},
        {'trigger': 'refinement', 'source': 'reference_and_secondary_disparity',
         'dest': 'reference_and_secondary_disparity', 'after': 'refinement_run_ref_and_sec'}
    ]

    _transitions_check = [
        {'trigger': 'stereo', 'source': 'begin', 'dest': 'cost_volume', 'after': 'stereo_check_conf'},
        {'trigger': 'aggregation', 'source': 'cost_volume', 'dest': 'cost_volume',
         'after': 'aggregation_check_conf'},
        {'trigger': 'optimization', 'source': 'cost_volume', 'dest': 'cost_volume',
         'after': 'optimization_check_conf'},
        {'trigger': 'disparity', 'source': 'cost_volume', 'dest': 'reference_disparity',
         'after': 'disparity_check_conf'},
        {'trigger': 'filter', 'source': 'reference_disparity', 'dest': 'reference_disparity',
         'after': 'filter_check_conf'},
        {'trigger': 'refinement', 'source': 'reference_disparity', 'dest': 'reference_disparity',
         'after': 'refinement_check_conf'},
        {'trigger': 'validation', 'source': 'reference_disparity', 'dest': 'reference_and_secondary_disparity',
         'after': 'validation_check_conf'},
        {'trigger': 'filter', 'source': 'reference_and_secondary_disparity',
         'dest': 'reference_and_secondary_disparity', 'after': 'filter_check_conf'},
        {'trigger': 'refinement', 'source': 'reference_and_secondary_disparity',
         'dest': 'reference_and_secondary_disparity',
         'after': 'refinement_check_conf'}
    ]

    def __init__(self, img_ref: xr.Dataset =None, img_sec: xr.Dataset =None,
                 disp_min: Union[int, np.ndarray] =None, disp_max: Union[int, np.ndarray] =None,
                 disp_min_sec: Union[int, np.ndarray] =None, disp_max_sec: Union[int, np.ndarray] =None):
        """
        Initialize Pandora Machine

        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param disp_min: minimal disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximal disparity
        :type disp_max: int or np.ndarray
        :param disp_min_sec: minimal disparity of the secondary image
        :type disp_min_sec: None, int or np.ndarray
        :param disp_max_sec: maximal disparity of the secondary image
        :type disp_max_sec: None, int or np.ndarray
        """

        self.img_ref = img_ref
        self.img_sec = img_sec
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.disp_min_sec = disp_min_sec
        self.disp_max_sec = disp_max_sec

        # Reference cost volume
        self.cv = None
        # Secondary cost volume
        self.cv_right = None
        # Reference disparity map
        self.ref_disparity = None
        # Secondary disparity map
        self.sec_disparity = None

        # Pandora's pipeline configuration
        self.pipeline_cfg = {'pipeline': {}}
        # Reference_pipeline = Pipeline for reference (left) disparity computation. False when it's about
        # secondary (right) disparity computation
        self.reference_pipeline = True
        # List of pandora steps that have been executed. Information to create the secondary disparity following the
        # same pipeline that has been used to create the reference one.
        self.steps_run = []

        # Define avalaible states
        states_ = ['begin', 'cost_volume', 'reference_disparity', 'reference_and_secondary_disparity', 'end']

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

        if self.reference_pipeline:
            self.cv = stereo_.compute_cost_volume(self.img_ref, self.img_sec, dmin_min, dmax_max, **cfg['image'])
            self.cv = stereo_.cv_masked(self.img_ref, self.img_sec, self.cv, self.disp_min, self.disp_max,
                                        **cfg['image'])

        else:
            if self.disp_min_sec is None:
                self.disp_min_sec = -self.disp_max
                self.disp_max_sec = -self.disp_min
            dmin_min_sec, dmax_max_sec = stereo_.dmin_dmax(self.disp_min_sec, self.disp_max_sec)
            self.cv_right = stereo_.compute_cost_volume(self.img_sec, self.img_ref, dmin_min_sec, dmax_max_sec,
                                                        **cfg['image'])
            self.cv_right = stereo_.cv_masked(self.img_sec, self.img_ref, self.cv_right, self.disp_min_sec,
                                              self.disp_max_sec, **cfg['image'])

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
        if self.reference_pipeline:
            self.ref_disparity = disparity.to_disp(self.cv, cfg['invalid_disparity'])
            self.ref_disparity = disparity.validity_mask(self.ref_disparity, self.img_ref, self.img_sec, self.cv,
                                                         **cfg['image'])
        else:
            self.sec_disparity = disparity.to_disp(self.cv_right, cfg['invalid_disparity'])
            self.sec_disparity = disparity.validity_mask(self.sec_disparity, self.img_sec, self.img_ref,
                                                         self.cv_right, **cfg['image'])

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
        if self.reference_pipeline:
            self.ref_disparity = filter_.filter_disparity(self.ref_disparity)
        else:
            self.sec_disparity = filter_.filter_disparity(self.sec_disparity)

    def filter_run_ref_and_sec(self, cfg: Dict[str, dict], input_step: str):
        """
        Disparity filter on reference and secondary disparity
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        logging.info('Disparity filtering Ref + Sec ...')
        filter_ = filter.AbstractFilter(**cfg['pipeline'][input_step])
        self.ref_disparity = filter_.filter_disparity(self.ref_disparity)
        self.sec_disparity = filter_.filter_disparity(self.sec_disparity)

    def refinement_run_ref_and_sec(self, cfg: Dict[str, dict], input_step: str):
        """
         Subpixel disparity refinement
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        refinement_ = refinement.AbstractRefinement(**cfg['pipeline'][input_step])
        logging.info('Subpixel refinement Ref + Sec ...')
        self.cv, self.ref_disparity = refinement_.subpixel_refinement(self.cv, self.ref_disparity)
        self.cv_right, self.sec_disparity = refinement_.subpixel_refinement(self.cv_right, self.sec_disparity)

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
        if self.reference_pipeline:
            self.cv, self.ref_disparity = refinement_.subpixel_refinement(self.cv, self.ref_disparity)
        else:
            self.cv_right, self.sec_disparity = refinement_.subpixel_refinement(self.cv_right, self.sec_disparity)

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
        if self.reference_pipeline:
            self.cv = aggregation_.cost_volume_aggregation(self.img_ref, self.img_sec, self.cv)
        else:
            self.cv_right = aggregation_.cost_volume_aggregation(self.img_sec, self.img_ref, self.cv_right)

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
        if self.reference_pipeline:
            self.cv = optimization_.optimize_cv(self.cv, self.img_ref, self.img_sec)
        else:
            self.cv_right = optimization_.optimize_cv(self.cv_right, self.img_sec, self.img_ref)

    def validation_run(self, cfg: Dict[str, dict], input_step: str):
        """
         Create reference and secondary disparity map. Computes the cross cheking. Interpolated mismatch and occlusions
        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return:
        """
        validation_ = validation.AbstractValidation(**cfg['pipeline'][input_step])
        if 'interpolated_disparity' in cfg:
            interpolate_ = validation.AbstractInterpolation(**cfg['pipeline'][input_step])
        logging.info('Computing the right disparity map with the accurate method...')

        # Create the secondary map with the same pipeline as the reference
        self.set_state('begin')
        self.reference_pipeline = False

        for e in self.steps_run:
            if e == 'validation':
                break
            self.run(e, cfg)

        # Apply cross checking
        self.ref_disparity = validation_.disparity_checking(self.ref_disparity, self.sec_disparity)
        self.sec_disparity = validation_.disparity_checking(self.sec_disparity, self.ref_disparity)

        # Interpolated mismatch and occlusions
        if 'interpolated_disparity' in cfg:
            self.ref_disparity = interpolate_.interpolated_disparity(self.ref_disparity)
            self.sec_disparity = interpolate_.interpolated_disparity(self.sec_disparity)

        self.set_state('reference_and_secondary_disparity')

    def run_prepare(self, img_ref: xr.Dataset, img_sec: xr.Dataset, disp_min: Union[int, np.ndarray],
                    disp_max: Union[int, np.ndarray], disp_min_sec: Union[None, int, np.ndarray] = None,
                    disp_max_sec: Union[None, int, np.ndarray] = None):
        """
        Prepare the machine before running

        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param disp_min: minimal disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximal disparity
        :type disp_max: int or np.ndarray
        :param disp_min_sec: minimal disparity of the secondary image
        :type disp_min_sec: None, int or np.ndarray
        :param disp_max_sec: maximal disparity of the secondary image
        :type disp_max_sec: None, int or np.ndarray
        """

        self.img_ref = img_ref
        self.img_sec = img_sec
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.disp_max_sec = disp_max_sec
        self.disp_min_sec = disp_min_sec
        self.ref_disparity = xr.Dataset()
        self.sec_disparity = xr.Dataset()
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
            self.steps_run.append(input_step)
            if len(input_step.split('_')) != 1:
                self.trigger(input_step.split('_')[0], cfg, input_step)
            else:
                self.trigger(input_step, cfg, input_step)
        except (MachineError, KeyError):
            raise MachineError

    def run_exit(self):

        self.remove_transitions(self._transitions_run)
        self.set_state('begin')
        self.reference_pipeline = True
        self.steps_run = []

    def stereo_check_conf(self, stereo_cfg: Dict[str, dict], input_step: str):
        """
        Check the stereo configuration

        :param stereo_cfg: stereo configuration
        :type stereo_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        stereo_ = stereo.AbstractStereo(**stereo_cfg)
        self.pipeline_cfg['pipeline'][input_step] = stereo_.cfg

    def disparity_check_conf(self, disparity_computation_cfg: Dict[str, dict], input_step: str):
        """
        Check the disparity computation configuration

        :param disparity_computation_cfg: disparity computation configuration
        :type disparity_computation_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        assert disparity_computation_cfg == 'wta'
        self.pipeline_cfg['pipeline'][input_step] = disparity_computation_cfg

    def filter_check_conf(self, filter_cfg: Dict[str, dict], input_step: str):
        """
        Check the filter configuration

        :param filter_cfg: filter configuration
        :type filter_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        filter_ = filter.AbstractFilter(**filter_cfg)
        self.pipeline_cfg['pipeline'][input_step] = filter_.cfg

    def refinement_check_conf(self, refinement_cfg: Dict[str, dict], input_step: str):
        """
        Check the refinement configuration

        :param refinement_cfg: refinement configuration
        :type refinement_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        refinement_ = refinement.AbstractRefinement(**refinement_cfg)
        self.pipeline_cfg['pipeline'][input_step] = refinement_.cfg

    def aggregation_check_conf(self, aggregation_cfg: Dict[str, dict], input_step: str):
        """
        Check the aggregation configuration

        :param aggregation_cfg: aggregation configuration
        :type aggregation_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        aggregation_ = aggregation.AbstractAggregation(**aggregation_cfg)
        self.pipeline_cfg['pipeline'][input_step] = aggregation_.cfg

    def optimization_check_conf(self, optimization_cfg: Dict[str, dict], input_step: str):
        """
        Check the optimization configuration

        :param optimization_cfg: optimization configuration
        :type optimization_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """
        optimization_ = optimization.AbstractOptimization(**optimization_cfg)
        self.pipeline_cfg['pipeline'][input_step] = optimization_.cfg

    def validation_check_conf(self, validation_cfg: Dict[str, dict], input_step: str):
        """
        Check the validation configuration

        :param validation_cfg: validation configuration
        :type validation_cfg: dict
        :param input_step: current step
        :type input_step: string
        :return:
        """

        validation_ = validation.AbstractValidation(**validation_cfg)
        self.pipeline_cfg['pipeline'][input_step] = validation_.cfg
        if 'interpolated_disparity' in validation_cfg:
            interpolate_ = validation.AbstractInterpolation(**validation_cfg)

    def check_conf(self, cfg: Dict[str, dict]):
        """
        Check configuration and transitions

        :param cfg: pipeline configuration
        :type  cfg: dict
        :return:
        """

        # Add transitions to the empty machine.
        self.add_transitions(self._transitions_check)

        for input_step in cfg:

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
                    self.trigger(input_step.split('_')[0], cfg[input_step], input_step)
                else:
                    self.trigger(input_step, cfg[input_step], input_step)

            except (MachineError, KeyError):
                print("MachineError")
                raise MachineError

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
