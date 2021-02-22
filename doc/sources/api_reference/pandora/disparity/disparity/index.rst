:mod:`pandora.disparity.disparity`
==================================

.. py:module:: pandora.disparity.disparity

.. autoapi-nested-parse::

   This module contains functions associated to the disparity map computation step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.disparity.disparity.AbstractDisparity
   pandora.disparity.disparity.WinnerTakesAll



.. py:class:: AbstractDisparity

   Abstract Disparity class

   .. attribute:: __metaclass__
      

      

   .. attribute:: disparity_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self)
      :abstractmethod:

      Describes the disparity method


   .. method:: to_disp(self, cv: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None) -> xr.Dataset
      :abstractmethod:

      Disparity computation by applying the Winner Takes All strategy

      :param cv: the cost volume datset with the data variables:

              - cv 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset,
      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :return: Dataset with the disparity map and the confidence measure with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :rtype: xarray.Dataset


   .. method:: coefficient_map(cv: xr.DataArray) -> xr.DataArray
      :staticmethod:

      Return the coefficient map

      :param cv: cost volume
      :type cv: xarray.Dataset, with the data variables cost_volume 3D xarray.DataArray (row, col, disp)
      :return: the coefficient map
      :rtype: 2D DataArray (row, col)


   .. method:: approximate_right_disparity(cv: xr.Dataset, img_right: xr.Dataset, invalid_value: float = 0) -> xr.Dataset
      :staticmethod:

      Create the right disparity map, by a diagonal search for the minimum in the left cost volume

      ERNST, Ines et HIRSCHMÜLLER, Heiko.
      Mutual information based semi-global stereo matching on the GPU.
      In : International Symposium on Visual Computing. Springer, Berlin, Heidelberg, 2008. p. 228-239.

      :param cv: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param invalid_value: disparity to assign to invalid pixels
      :type invalid_value: float
      :return: Dataset with the right disparity map, the confidence measure and the validity mask with         the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
              - validity_mask 2D xarray.DataArray (row, col)
      :rtype: xarray.Dataset


   .. method:: validity_mask(self, disp: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset) -> None

      Create the validity mask of the disparity map

      :param disp: dataset with the disparity map and the confidence measure
      :type disp: xarray.Dataset with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cv: cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :return: None


   .. method:: mask_border(disp: xr.Dataset)
      :staticmethod:

      Mask border pixel  which haven't been calculated because of the window's size

      :param disp: dataset with the disparity map and the confidence measure  with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :type disp: xarray.Dataset
      :return: None


   .. method:: mask_invalid_variable_disparity_range(disp, cv) -> None
      :staticmethod:

      Mask the pixels that have a missing disparity range, searching in the cost volume
      the pixels where cost_volume(row,col, for all d) = np.nan

      :param disp: dataset with the disparity map and the confidence measure with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :type disp: xarray.Dataset
      :param cv: cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :return: None


   .. method:: allocate_left_mask(disp: xr.Dataset, img_left: xr.Dataset) -> None
      :staticmethod:

      Allocate the left image mask

      :param disp: dataset with the disparity map and the confidence measure with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :type disp: xarray.Dataset
      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :return: None


   .. method:: allocate_right_mask(disp: xr.Dataset, img_right: xr.Dataset, bit_1: Union[np.ndarray, Tuple]) -> None
      :staticmethod:

      Allocate the right image mask

      :param disp: dataset with the disparity map and the confidence measure with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray(row, col, indicator)
      :type disp: xarray.Dataset
      :param img_right: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param bit_1: where the disparity interval is missing in the right image ( disparity range outside the image )
      :type: ndarray or Tuple
      :return: None



.. py:class:: WinnerTakesAll(**cfg)

   Bases: :class:`pandora.disparity.disparity.AbstractDisparity`

   WinnerTakesAll class allows to perform the disparity computation step

   .. attribute:: _INVALID_DISPARITY
      

      

   .. method:: check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: disparity configuration
      :type cfg: dict
      :return cfg: disparity configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the disparity method
      :return: None


   .. method:: to_disp(self, cv: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None) -> xr.Dataset

      Disparity computation by applying the Winner Takes All strategy

      :param cv: the cost volume datset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :return: Dataset with the disparity map and the confidence measure  with the data variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :rtype: xarray.Dataset


   .. method:: argmin_split(cost_volume: xr.Dataset) -> np.ndarray
      :staticmethod:

      Find the indices of the minimum values for a 3D DataArray, along axis 2.
      Memory consumption is reduced by splitting the 3D Array.

      :param cost_volume: the cost volume dataset
      :type cost_volume: xarray.Dataset
      :return: the disparities for which the cost volume values are the smallest
      :rtype: np.ndarray


   .. method:: argmax_split(cost_volume: xr.Dataset) -> np.ndarray
      :staticmethod:

      Find the indices of the maximum values for a 3D DataArray, along axis 2.
      Memory consumption is reduced by splitting the 3D Array.

      :param cost_volume: the cost volume dataset
      :type cost_volume: xarray.Dataset
      :return: the disparities for which the cost volume values are the highest
      :rtype: np.ndarray



