:mod:`pandora.cost_volume_confidence.std_intensity`
===================================================

.. py:module:: pandora.cost_volume_confidence.std_intensity

.. autoapi-nested-parse::

   This module contains functions for estimating confidence from image.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.cost_volume_confidence.std_intensity.StdIntensity



.. py:class:: StdIntensity(**cfg: str)

   Bases: :class:`pandora.cost_volume_confidence.cost_volume_confidence.AbstractCostVolumeConfidence`

   StdIntensity class allows to estimate a confidence measure from the left image by calculating the standard
   deviation of the intensity

   .. method:: check_conf(**cfg: str) -> Dict[str, str]
      :staticmethod:

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: std_intensity configuration
      :type cfg: dict
      :return cfg: std_intensity configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the confidence method
      :return: None


   .. method:: confidence_prediction(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> Tuple[xr.Dataset, xr.Dataset]

      Computes a confidence measure that evaluates the standard deviation of intensity of the left image

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



