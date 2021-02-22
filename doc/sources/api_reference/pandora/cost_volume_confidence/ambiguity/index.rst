:mod:`pandora.cost_volume_confidence.ambiguity`
===============================================

.. py:module:: pandora.cost_volume_confidence.ambiguity

.. autoapi-nested-parse::

   This module contains functions for estimating confidence from ambiguity.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.cost_volume_confidence.ambiguity.Ambiguity



.. py:class:: Ambiguity(**cfg: str)

   Bases: :class:`pandora.cost_volume_confidence.cost_volume_confidence.AbstractCostVolumeConfidence`

   Ambiguity class allows to estimate a confidence from the cost volume

   .. attribute:: _ETA_MIN
      :annotation: = 0.0

      

   .. attribute:: _ETA_MAX
      :annotation: = 0.7

      

   .. attribute:: _ETA_STEP
      :annotation: = 0.01

      

   .. attribute:: _PERCENTILE
      :annotation: = 1.0

      

   .. method:: check_conf(self, **cfg: str) -> Dict[str, str]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: ambiguity configuration
      :type cfg: dict
      :return cfg: ambiguity configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the confidence method
      :return: None


   .. method:: confidence_prediction(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> Tuple[xr.Dataset, xr.Dataset]

      Computes a confidence measure that evaluates the matching cost function at each point

      :param disp: the disparity map dataset
      :type disp: xarray.Dataset
      :param img_left: left Dataset image
      :tye img_left: xarray.Dataset
      :param img_right: right Dataset image
      :type img_right: xarray.Dataset
      :param cv: cost volume dataset
      :type cv: xarray.Dataset
      :return: the disparity map and the cost volume with a new indicator 'ambiguity_confidence' in the DataArray
               confidence_measure
      :rtype: Tuple(xarray.Dataset, xarray.Dataset) with the data variables:

              - confidence_measure 3D xarray.DataArray (row, col, indicator)


   .. method:: normalize_with_percentile(self, ambiguity)

      Normalize ambiguity with percentile

      :param ambiguity: ambiguity
      :type ambiguity: 2D np.array (row, col) dtype = float32
      :return: the normalized ambiguity
      :rtype: 2D np.array (row, col) dtype = float32


   .. method:: compute_ambiguity(cv: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float) -> np.ndarray
      :staticmethod:

      Computes ambiguity.

      :param cv: cost volume
      :type cv: 3D np.array (row, col, disp)
      :param _eta_min: minimal eta
      :type _eta_min: float
      :param _eta_max: maximal eta
      :type _eta_max: float
      :param _eta_step: eta step
      :type _eta_step: float
      :return: the normalized ambiguity
      :rtype: 2D np.array (row, col) dtype = float32


   .. method:: compute_ambiguity_and_sampled_ambiguity(cv: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float)
      :staticmethod:

      Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks

      :param cv: cost volume
      :type cv: 3D np.array (row, col, disp)
      :param _eta_min: minimal eta
      :type _eta_min: float
      :param _eta_max: maximal eta
      :type _eta_max: float
      :param _eta_step: eta step
      :type _eta_step: float
      :return: the normalized ambiguity and sampled ambiguity
      :rtype: Tuple(2D np.array (row, col) dtype = float32, 3D np.array (row, col) dtype = float32)



