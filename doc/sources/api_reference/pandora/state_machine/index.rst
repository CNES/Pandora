:mod:`pandora.state_machine`
============================

.. py:module:: pandora.state_machine

.. autoapi-nested-parse::

   This module contains class associated to the pandora state machine



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.state_machine.PandoraMachine



.. py:class:: PandoraMachine(img_left_pyramid: List[xr.Dataset] = None, img_right_pyramid: List[xr.Dataset] = None, disp_min: Union[np.array, int] = None, disp_max: Union[np.array, int] = None, right_disp_min: Union[np.array, None] = None, right_disp_max: Union[np.array, None] = None)

   Bases: :class:`transitions.extensions.GraphMachine`

   PandoraMachine class to create and use a state machine

   .. attribute:: _transitions_run
      

      

   .. attribute:: _transitions_check
      

      

   .. method:: matching_cost_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Matching cost computation
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: aggregation_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Cost (support) aggregation
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: optimization_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Cost optimization
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: disparity_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Disparity computation and validity mask
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: filter_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Disparity filter
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: refinement_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Subpixel disparity refinement
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: validation_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Validation of disparity map
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: run_multiscale(self, cfg: Dict[str, dict], input_step: str) -> None

      Compute the disparity range for the next scale
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: cost_volume_confidence_run(self, cfg: Dict[str, dict], input_step: str) -> None

      Confidence prediction
      :param cfg: pipeline configuration
      :type  cfg: dict
      :param input_step: step to trigger
      :type input_step: str
      :return: None


   .. method:: run_prepare(self, cfg: Dict[str, dict], left_img: xr.Dataset, right_img: xr.Dataset, disp_min: Union[np.array, int], disp_max: Union[np.array, int], scale_factor: Union[None, int] = None, num_scales: Union[None, int] = None, right_disp_min: Union[None, np.array] = None, right_disp_max: Union[None, np.array] = None) -> None

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


   .. method:: run(self, input_step: str, cfg: Dict[str, dict]) -> None

      Run pandora step by triggering the corresponding machine transition

      :param input_step: step to trigger
      :type input_step: str
      :param cfg: pipeline configuration
      :type  cfg: dict
      :return: None


   .. method:: run_exit(self) -> None

      Clear transitions and return to state begin

      :return: None


   .. method:: right_disp_map_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the right_disp_map configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the matching cost configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: disparity_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the disparity computation configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: filter_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the filter configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: refinement_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the refinement configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: aggregation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the aggregation configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: optimization_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the optimization configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: validation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the validation configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: multiscale_check_conf(self, cfg: Dict[str, dict], input_step: str)

      Check the disparity computation configuration

      :param cfg: disparity computation configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return:


   .. method:: cost_volume_confidence_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None

      Check the confidence configuration

      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: None


   .. method:: check_conf(self, cfg: Dict[str, dict])

      Check configuration and transitions

      :param cfg: pipeline configuration
      :type  cfg: dict
      :return:


   .. method:: remove_transitions(self, transition_list: Dict[str, dict]) -> None

      Delete all transitions defined in the input list

      :param transition_list: list of transitions
      :type transition_list: dict
      :return: None


   .. method:: is_not_last_scale(self, input_step: str, cfg: Dict[str, dict])

      Check if the current scale is the last scale
      :param cfg: configuration
      :type cfg: dict
      :param input_step: current step
      :type input_step: string
      :return: boolean



