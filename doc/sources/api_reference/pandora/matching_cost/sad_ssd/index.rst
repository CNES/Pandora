:mod:`pandora.matching_cost.sad_ssd`
====================================

.. py:module:: pandora.matching_cost.sad_ssd

.. autoapi-nested-parse::

   This module contains functions associated to SAD and SSD methods used in the cost volume measure step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.matching_cost.sad_ssd.SadSsd



.. py:class:: SadSsd(**cfg: Union[str, int])

   Bases: :class:`pandora.matching_cost.matching_cost.AbstractMatchingCost`

   SadSsd class allows to compute the cost volume

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


   .. method:: ad_cost(point_p: Tuple[int, int], point_q: Tuple[int, int], img_left: xr.Dataset, img_right: xr.Dataset) -> np.ndarray
      :staticmethod:

      Computes the absolute difference

      :param point_p: Point interval, in the left image, over which the squared difference will be applied
      :type point_p: tuple
      :param point_q: Point interval, in the right image, over which the squared difference will be applied
      :type point_q: tuple
      :param img_left: left Dataset image
      :type img_left:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :param img_right: right Dataset image
      :type img_right:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :return: the absolute difference pixel-wise between elements in the interval
      :rtype: numpy array


   .. method:: sd_cost(point_p: Tuple, point_q: Tuple, img_left: xr.Dataset, img_right: xr.Dataset) -> np.ndarray
      :staticmethod:

      Computes the square difference

      :param point_p: Point interval, in the left image, over which the squared difference will be applied
      :type point_p: tuple
      :param point_q: Point interval, in the right image, over which the squared difference will be applied
      :type point_q: tuple
      :param img_left: left Dataset image
      :type img_left:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :param img_right: right Dataset image
      :type img_right:
          xarray.Dataset containing :
              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :return: the squared difference pixel-wise between elements in the interval
      :rtype: numpy array


   .. method:: pixel_wise_aggregation(self, cost_volume: np.ndarray) -> np.ndarray

      Summing pixel wise matching cost over square windows

      :param cost_volume: the cost volume
      :type cost_volume: numpy array 3D (disp, col, row)
      :return: the cost volume aggregated
      :rtype: numpy array 3D ( disp, col, row)



