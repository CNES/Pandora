:mod:`pandora`
==============

.. py:module:: pandora

.. autoapi-nested-parse::

   This module contains functions to run Pandora pipeline.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   aggregation/index.rst
   cost_volume_confidence/index.rst
   disparity/index.rst
   filter/index.rst
   matching_cost/index.rst
   multiscale/index.rst
   optimization/index.rst
   refinement/index.rst
   validation/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   Pandora/index.rst
   check_json/index.rst
   common/index.rst
   constants/index.rst
   img_tools/index.rst
   marge/index.rst
   output_tree_design/index.rst
   state_machine/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pandora.run
   pandora.setup_logging
   pandora.import_plugin
   pandora.main


.. function:: run(pandora_machine: PandoraMachine, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: Union[np.array, int], disp_max: Union[np.array, int], cfg: Dict[str, dict], disp_min_right: Union[None, np.array] = None, disp_max_right: Union[None, np.array] = None) -> Tuple[xr.Dataset, xr.Dataset]

   Run the pandora pipeline

   :param pandora_machine: instance of PandoraMachine
   :type pandora_machine: PandoraMachine
   :param img_left: left Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
           - msk (optional): 2D (row, col) xarray.DataArray
   :type img_left: xarray.Dataset
   :param img_right: right Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
           - msk (optional): 2D (row, col) xarray.DataArray
   :type img_right: xarray.Dataset
   :param disp_min: minimal disparity
   :type disp_min: int or np.array
   :param disp_max: maximal disparity
   :type disp_max: int or np.array
   :param cfg: pipeline configuration
   :type cfg: Dict[str, dict]
   :param disp_min_right: minimal disparity of the right image
   :type disp_min_right: np.array or None
   :param disp_max_right: maximal disparity of the right image
   :type disp_max_right: np.array or None
   :return: Two xarray.Dataset :


           - left : the left dataset, which contains the variables :
               - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
               - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray                     (row, col, indicator)
               - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

           - right : the right dataset. If there is no validation step, the right Dataset will be empty.If a             validation step is configured, the dataset will contain the variables :
               - disparity_map : the disparity map in the geometry of the right image 2D DataArray (row, col)
               - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray                     (row, col, indicator)
               - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

   :rtype: tuple (xarray.Dataset, xarray.Dataset)


.. function:: setup_logging(verbose: bool) -> None

   Setup the logging configuration

   :param verbose: verbose mode
   :type verbose: bool
   :return: None


.. function:: import_plugin() -> None

   Load all the registered entry points
   :return: None


.. function:: main(cfg_path: str, output: str, verbose: bool) -> None

   Check config file and run pandora framework accordingly

   :param cfg_path: path to the json configuration file
   :type cfg_path: string
   :param output: Path to output directory
   :type output: string
   :param verbose: verbose mode
   :type verbose: bool
   :return: None


