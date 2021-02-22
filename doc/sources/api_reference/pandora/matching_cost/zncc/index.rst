:mod:`pandora.matching_cost.zncc`
=================================

.. py:module:: pandora.matching_cost.zncc

.. autoapi-nested-parse::

   This module contains functions associated to ZNCC method used in the cost volume measure step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.matching_cost.zncc.Zncc



.. py:class:: Zncc(**cfg: Union[str, int])

   Bases: :class:`pandora.matching_cost.matching_cost.AbstractMatchingCost`

   Zero mean normalized cross correlation
   Zncc class allows to compute the cost volume

   .. attribute:: _WINDOW_SIZE
      :annotation: = 5

      

   .. attribute:: _SUBPIX
      :annotation: = 1

      

   .. method:: check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: matching cost configuration
      :type cfg: dict
      :return cfg: matching cost configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the matching cost method
      :return: None


   .. method:: compute_cost_volume(self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int) -> xr.Dataset

      Computes the cost volume for a pair of images

      :param img_left: left Dataset image
      :type img_left:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :param img_right: right Dataset image
      :type img_right:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :param disp_min: minimum disparity
      :type disp_min: int
      :param disp_max: maximum disparity
      :type disp_max: int
      :return: the cost volume dataset
      :rtype:
          xarray.Dataset, with the data variables:
              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)



