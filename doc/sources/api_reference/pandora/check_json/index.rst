:mod:`pandora.check_json`
=========================

.. py:module:: pandora.check_json

.. autoapi-nested-parse::

   This module contains functions allowing to check the configuration given to Pandora pipeline.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pandora.check_json.rasterio_can_open_mandatory
   pandora.check_json.rasterio_can_open
   pandora.check_json.check_images
   pandora.check_json.check_disparities
   pandora.check_json.get_config_pipeline
   pandora.check_json.get_config_input
   pandora.check_json.check_pipeline_section
   pandora.check_json.check_input_section
   pandora.check_json.check_conf
   pandora.check_json.concat_conf
   pandora.check_json.read_multiscale_params
   pandora.check_json.read_config_file
   pandora.check_json.update_conf


.. function:: rasterio_can_open_mandatory(file_: str) -> bool

   Test if file can be open by rasterio

   :param file_: File to test
   :type file_: string
   :returns: True if rasterio can open file and False otherwise
   :rtype: bool


.. function:: rasterio_can_open(file_: str) -> bool

   Test if file can be open by rasterio

   :param file_: File to test
   :type file_: string
   :returns: True if rasterio can open file and False otherwise
   :rtype: bool


.. function:: check_images(img_left: str, img_right: str, msk_left: str, msk_right: str) -> None

   Check the images

   :param img_left: path to the left image
   :type img_left: string
   :param img_right: path to the right image
   :type img_right: string
   :param msk_left: path to the mask of the left image
   :type msk_left: string
   :param msk_right: path to the mask of the right image
   :type msk_right: string
   :return: None


.. function:: check_disparities(disp_min: Union[int, str, None], disp_max: Union[int, str, None], right_disp_min: Union[str, None], right_disp_max: Union[str, None], img_left: str) -> None

   Check left and right disparities.

   :param disp_min: minimal disparity
   :type disp_min: int or str or None
   :param disp_max: maximal disparity
   :type disp_max: int or str or None
   :param right_disp_min: right minimal disparity
   :type right_disp_min: str or None
   :param right_disp_max: right maximal disparity
   :type right_disp_max: str or None
   :param img_left: path to the left image
   :type img_left: str
   :return: None


.. function:: get_config_pipeline(user_cfg: Dict[str, dict]) -> Dict[str, dict]

   Get the pipeline configuration

   :param user_cfg: user configuration
   :type user_cfg: dict
   :return: cfg: partial configuration
   :rtype: cfg: dict


.. function:: get_config_input(user_cfg: Dict[str, dict]) -> Dict[str, dict]

   Get the input configuration

   :param user_cfg: user configuration
   :type user_cfg: dict
   :return cfg: partial configuration
   :rtype cfg: dict


.. function:: check_pipeline_section(user_cfg: Dict[str, dict], pandora_machine: PandoraMachine) -> Dict[str, dict]

   Check if the pipeline is correct by
   - Checking the sequence of steps according to the machine transitions
   - Checking parameters, define in dictionary, of each Pandora step

   :param user_cfg: pipeline user configuration
   :type user_cfg: dict
   :param pandora_machine: instance of PandoraMachine
   :type pandora_machine: PandoraMachine object
   :return: cfg: pipeline configuration
   :rtype: cfg: dict


.. function:: check_input_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]

   Complete and check if the dictionary is correct

   :param user_cfg: user configuration
   :type user_cfg: dict
   :return: cfg: global configuration
   :rtype: cfg: dict


.. function:: check_conf(user_cfg: Dict[str, dict], pandora_machine: PandoraMachine) -> dict

   Complete and check if the dictionary is correct

   :param user_cfg: user configuration
   :type user_cfg: dict
   :param pandora_machine: instance of PandoraMachine
   :type pandora_machine: PandoraMachine
   :return: cfg: global configuration
   :rtype: cfg: dict


.. function:: concat_conf(cfg_list: List[Dict[str, dict]]) -> Dict[str, dict]

   Concatenate dictionaries

   :param cfg_list: list of configurations
   :type cfg_list: List of dict
   :return: cfg: global configuration
   :rtype: cfg: dict


.. function:: read_multiscale_params(cfg: Dict[str, dict]) -> Tuple[int, int]

   Returns the multiscale parameters

   :param cfg: configuration
   :type cfg: dict
   :return:
       - num_scales: number of scales
       - scale_factor: factor by which each coarser layer is downsampled
   :rtype: tuple(int, int )


.. data:: input_configuration_schema
   

   

.. data:: input_configuration_schema_integer_disparity
   

   

.. data:: input_configuration_schema_left_disparity_grids_right_none
   

   

.. data:: input_configuration_schema_left_disparity_grids_right_grids
   

   

.. data:: default_short_configuration_input
   

   

.. data:: default_short_configuration_pipeline
   

   

.. data:: default_short_configuration
   

   

.. function:: read_config_file(config_file: str) -> Dict[str, dict]

   Read a json configuration file

   :param config_file: path to a json file containing the algorithm parameters
   :type config_file: string
   :return user_cfg: configuration dictionary
   :rtype: dict


.. function:: update_conf(def_cfg: Dict[str, dict], user_cfg: Dict[str, dict]) -> Dict[str, dict]

   Update the default configuration with the user configuration,

   :param def_cfg: default configuration
   :type def_cfg: dict
   :param user_cfg: user configuration
   :type user_cfg: dict
   :return: the user and default configuration
   :rtype: dict


