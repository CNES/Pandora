:mod:`pandora.filter.bilateral`
===============================

.. py:module:: pandora.filter.bilateral

.. autoapi-nested-parse::

   This module contains functions associated to the bilateral filter used to filter the disparity map.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.filter.bilateral.BilateralFilter



.. py:class:: BilateralFilter(**cfg: Union[str, float])

   Bases: :class:`pandora.filter.filter.AbstractFilter`

   BilateralFilter class allows to perform the filtering step

   .. attribute:: _SIGMA_COLOR
      :annotation: = 2.0

      

   .. attribute:: _SIGMA_SPACE
      :annotation: = 6.0

      

   .. method:: check_conf(self, **cfg: Union[str, float]) -> Dict[str, Union[str, float]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: filter configuration
      :type cfg: dict
      :return cfg: filter configuration updated
      :rtype: dict


   .. method:: desc(self)

      Describes the filtering method


   .. method:: filter_disparity(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None

      Apply bilateral filter using openCV.
      Filter size is computed from sigmaSpace

      :param disp: the disparity map dataset  with the variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
              - validity_mask 2D xarray.DataArray (row, col)
      :type disp: xarray.Dataset
      :param img_left: left Dataset image
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image
      :type img_right: xarray.Dataset
      :param cv: cost volume dataset
      :type cv: xarray.Dataset
      :return: None



