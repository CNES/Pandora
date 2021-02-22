:mod:`pandora.multiscale.fixed_zoom_pyramid`
============================================

.. py:module:: pandora.multiscale.fixed_zoom_pyramid

.. autoapi-nested-parse::

   This module contains functions associated to the multi-scale pyramid method.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.multiscale.fixed_zoom_pyramid.FixedZoomPyramid



.. py:class:: FixedZoomPyramid(**cfg: dict)

   Bases: :class:`pandora.multiscale.multiscale.AbstractMultiscale`

   FixedZoomPyramid class, allows to perform the multiscale processing

   .. attribute:: _PYRAMID_NUM_SCALES
      :annotation: = 2

      

   .. attribute:: _PYRAMID_SCALE_FACTOR
      :annotation: = 2

      

   .. attribute:: _PYRAMID_MARGE
      :annotation: = 1

      

   .. method:: check_conf(self, **cfg: Union[str, float, int]) -> Dict[str, Union[str, float, int]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: aggregation configuration
      :type cfg: dict
      :return cfg: aggregation configuration updated
      :rtype: dict


   .. method:: desc(self)

      Describes the aggregation method


   .. method:: disparity_range(self, disp: xr.Dataset, disp_min: int, disp_max: int) -> Tuple[np.array, np.array]

      Disparity range computation by seeking the max and min values in the window.
      Invalid disparities are given the full disparity range

      :param disp: the disparity dataset
      :type disp: xarray.Dataset with the data variables :
              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :param disp_min: absolute min disparity
      :type disp_min: int
      :param disp_max: absolute max disparity
      :type disp_max: int
      :return: Two np.darray :
              - disp_min_range : minimum disparity value for all pixels.
              - disp_max_range : maximum disparity value for all pixels.

      :rtype: tuple (np.ndarray, np.ndarray)



