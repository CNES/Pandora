:mod:`pandora.validation.interpolated_disparity`
================================================

.. py:module:: pandora.validation.interpolated_disparity

.. autoapi-nested-parse::

   This module contains classes and functions associated to the interpolation of the disparity map for the validation step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.validation.interpolated_disparity.AbstractInterpolation
   pandora.validation.interpolated_disparity.McCnnInterpolation
   pandora.validation.interpolated_disparity.SgmInterpolation



.. py:class:: AbstractInterpolation

   Abstract Interpolation class

   .. attribute:: __metaclass__
      

      

   .. attribute:: interpolation_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self) -> None
      :abstractmethod:

      Describes the disparity interpolation method for the validation step
      :return: None


   .. method:: interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None
      :abstractmethod:

      Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

      :param left: left Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type left: xarray.Dataset
      :param img_left: left Datset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cv: cost_volume Dataset with the variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :return: None



.. py:class:: McCnnInterpolation(**cfg: dict)

   Bases: :class:`pandora.validation.interpolated_disparity.AbstractInterpolation`

   McCnnInterpolation class allows to perform the interpolation of the disparity map

   .. method:: check_config(self, **cfg: dict) -> None

      Check and update the configuration

      :param cfg: optional configuration, {}
      :type cfg: dictionary
      :return: None


   .. method:: desc(self) -> None

      Describes the disparity interpolation method
      :return: None


   .. method:: interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None

      Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

      :param left: left Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type left: xarray.Dataset
      :param img_left: left Datset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cv: cost_volume Dataset with the variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :return: None


   .. method:: interpolate_occlusion_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
      :staticmethod:

      Interpolation of the left disparity map to resolve occlusion conflicts.
      Interpolate occlusion by moving left until
      we find a position labeled correct.

      Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
      patches. The journal of machine learning research, 17(1), 2287-2318.

      :param disp: disparity map
      :type disp: 2D np.array (row, col)
      :param valid: validity mask
      :type valid: 2D np.array (row, col)
      :return: the interpolate left disparity map, with the validity mask update :

          - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
      :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))


   .. method:: interpolate_mismatch_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
      :staticmethod:

      Interpolation of the left disparity map to resolve mismatch conflicts.
      Interpolate mismatch by finding the nearest
      correct pixels in 16 different directions and use the median of their disparities.

      Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
      patches. The journal of machine learning research, 17(1), 2287-2318.

      :param disp: disparity map
      :type disp: 2D np.array (row, col)
      :param valid: validity mask
      :type valid: 2D np.array (row, col)
      :return: the interpolate left disparity map, with the validity mask update :

          - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
      :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))



.. py:class:: SgmInterpolation(**cfg: dict)

   Bases: :class:`pandora.validation.interpolated_disparity.AbstractInterpolation`

   SgmInterpolation class allows to perform the interpolation of the disparity map

   .. method:: check_config(self, **cfg: dict) -> None

      Check and update the configuration

      :param cfg: optional configuration, {}
      :type cfg: dictionary
      :return: None


   .. method:: desc(self) -> None

      Describes the disparity interpolation method
      :return: None


   .. method:: interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None

      Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

      :param left: left Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type left: xarray.Dataset
      :param img_left: left Datset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cv: cost_volume Dataset with the variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :return: None


   .. method:: interpolate_occlusion_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
      :staticmethod:

      Interpolation of the left disparity map to resolve occlusion conflicts.
      Interpolate occlusion by moving by selecting
      the right lowest value along paths from 8 directions.

      HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
      IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

      :param disp: disparity map
      :type disp: 2D np.array (row, col)
      :param valid: validity mask
      :type valid: 2D np.array (row, col)
      :return: the interpolate left disparity map, with the validity mask update :

          - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
      :rtype: : tuple(2D np.array (row, col), 2D np.array (row, col))


   .. method:: interpolate_mismatch_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
      :staticmethod:

      Interpolation of the left disparity map to resolve mismatch conflicts. Interpolate mismatch by finding the
      nearest correct pixels in 8 different directions and use the median of their disparities.
      Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions.

      HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
      IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

      :param disp: disparity map
      :type disp: 2D np.array (row, col)
      :param valid: validity mask
      :type valid: 2D np.array (row, col)
      :return: the interpolate left disparity map, with the validity mask update :

          - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
      :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))



