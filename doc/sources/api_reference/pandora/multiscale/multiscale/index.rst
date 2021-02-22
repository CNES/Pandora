:mod:`pandora.multiscale.multiscale`
====================================

.. py:module:: pandora.multiscale.multiscale

.. autoapi-nested-parse::

   This module contains functions associated to the multiscale step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.multiscale.multiscale.AbstractMultiscale



.. py:class:: AbstractMultiscale

   Abstract Multiscale class

   .. attribute:: __metaclass__
      

      

   .. attribute:: multiscale_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str, *args)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string
      :param args: allows to register one plugin that contains different methods


   .. method:: desc(self)
      :abstractmethod:

      Describes the multiscale method


   .. method:: disparity_range(self, disp: xr.Dataset, disp_min: int, disp_max: int) -> Tuple[np.array, np.array]
      :abstractmethod:

      Disparity range computation by seeking the max and min values in the window.
      Unvalid disparities are given the full disparity range

      :param disp: the disparity dataset
      :type disp: xarray.Dataset with the data variables :
              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :param disp_min: absolute min disparity
      :type disp_min: int
      :param disp_max: absolute max disparity
      :type disp_max: int
      :return: Two np.darray :
              - disp_min_range : minimum disparity value for all pixels :
              - disp_max_range : maximum disparity value for all pixels.

      :rtype: tuple (np.ndarray, np.ndarray)


   .. method:: mask_invalid_disparities(disp: xr.Dataset) -> np.ndarray
      :staticmethod:

      Return a copied disparity map with all invalid disparities set to Nan

      :param disp: the disparity dataset
      :type disp: xarray.Dataset with the data variables :
              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)

      :return: np.darray :
              - filtered_disp_map : disparity map with invalid values set to Nzn

      :rtype: tuple (np.ndarray, np.ndarray)



