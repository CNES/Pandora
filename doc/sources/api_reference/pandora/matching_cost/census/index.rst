:mod:`pandora.matching_cost.census`
===================================

.. py:module:: pandora.matching_cost.census

.. autoapi-nested-parse::

   This module contains functions associated to census method used in the cost volume measure step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.matching_cost.census.Census



.. py:class:: Census(**cfg: Dict[str, Union[str, int]])

   Bases: :class:`pandora.matching_cost.matching_cost.AbstractMatchingCost`

   Census class allows to compute the cost volume

   .. attribute:: _WINDOW_SIZE
      :annotation: = 5

      

   .. attribute:: _SUBPIX
      :annotation: = 1

      

   .. method:: check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]

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

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param disp_min: minimum disparity
      :type disp_min: int
      :param disp_max: maximum disparity
      :type disp_max: int
      :return: the cost volume dataset , with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :rtype: xarray.Dataset


   .. method:: census_cost(self, point_p: Tuple[int, int], point_q: Tuple[int, int], img_left: xr.Dataset, img_right: xr.Dataset) -> List[int]

      Computes xor pixel-wise between pre-processed images by census transform

      :param point_p: Point interval, in the left image, over which the squared difference will be applied
      :type point_p: tuple
      :param point_q: Point interval, in the right image, over which the squared difference will be applied
      :type point_q: tuple
      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :return: the xor pixel-wise between elements in the interval
      :rtype: numpy array


   .. method:: popcount32b(row: int) -> int
      :staticmethod:

      Computes the Hamming weight for the input row,
      Hamming weight is the number of symbols that are different from the zero

      :param row: 32-bit integer
      :type row: int
      :return: the number of symbols that are different from the zero
      :rtype: int



