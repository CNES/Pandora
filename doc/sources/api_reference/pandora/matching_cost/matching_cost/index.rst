:mod:`pandora.matching_cost.matching_cost`
==========================================

.. py:module:: pandora.matching_cost.matching_cost

.. autoapi-nested-parse::

   This module contains functions associated to the cost volume measure step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.matching_cost.matching_cost.AbstractMatchingCost



.. py:class:: AbstractMatchingCost

   Abstract Matching Cost class

   .. attribute:: __metaclass__
      

      

   .. attribute:: matching_cost_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str, *args)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string
      :param args: allows to register one plugin that contains different methods


   .. method:: desc(self) -> None
      :abstractmethod:

      Describes the matching cost method
      :return: None


   .. method:: compute_cost_volume(self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int) -> xr.Dataset
      :abstractmethod:

      Computes the cost volume for a pair of images

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset  containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param disp_min: minimum disparity
      :type disp_min: int
      :param disp_max: maximum disparity
      :type disp_max: int
      :return: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :rtype: xarray.Dataset


   .. method:: allocate_costvolume(img_left: xr.Dataset, subpix: int, disp_min: int, disp_max: int, window_size: int, metadata: dict, np_data: np.ndarray = None) -> xr.Dataset
      :staticmethod:

      Allocate the cost volume

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param subpix: subpixel precision = (1 or 2 or 4)
      :type subpix: int
      :param disp_min: minimum disparity
      :type disp_min: int
      :param disp_max: maximum disparity
      :type disp_max: int
      :param window_size: size of the aggregation window
      :type window_size: int, odd number
      :param metadata: dictionary storing arbitrary metadata
      :type metadata: dictionary
      :param np_data: the array’s data
      :type np_data: 3D numpy array, dtype=np.float32
      :return: the dataset cost volume with the cost_volume and the confidence measure with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
      :rtype: xarray.Dataset


   .. method:: point_interval(img_left: xr.Dataset, img_right: xr.Dataset, disp: float) -> Tuple[Tuple[int, int], Tuple[int, int]]
      :staticmethod:

      Computes the range of points over which the similarity measure will be applied

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param disp: current disparity
      :type disp: float
      :return: the range of the left and right image over which the similarity measure will be applied
      :rtype: tuple


   .. method:: masks_dilatation(img_left: xr.Dataset, img_right: xr.Dataset, window_size: int, subp: int) -> Tuple[xr.DataArray, List[xr.DataArray]]
      :staticmethod:

      Return the left and right mask with the convention :
          - Invalid pixels are nan
          - No_data pixels are nan
          - Valid pixels are 0

      Apply dilation on no_data : if a pixel contains a no_data in its aggregation window, then the central pixel
      becomes no_data

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param window_size: window size of the measure
      :type window_size: int
      :param subp: subpixel precision = (1 or 2 or 4)
      :type subp: int
      :return: the left mask and the right masks:

              - left mask :  xarray.DataArray msk 2D(row, col)
              - right mask :  xarray.DataArray msk 2D(row, col)
              - right mask shifted :  xarray.DataArray msk 2D(row, shifted col by 0.5)
      :rtype: tuple (left mask, list[right mask, right mask shifted by 0.5])


   .. method:: dmin_dmax(disp_min: Union[int, np.ndarray], disp_max: Union[int, np.ndarray]) -> Tuple[int, int]
      :staticmethod:

      Find the smallest disparity present in disp_min, and the highest disparity present in disp_max

      :param disp_min: minimum disparity
      :type disp_min: int or np.ndarray
      :param disp_max: maximum disparity
      :type disp_max: int or np.ndarray
      :return: dmin_min: the smallest disparity in disp_min, dmax_max: the highest disparity in disp_max
      :rtype: Tuple(int, int)


   .. method:: cv_masked(self, img_left: xr.Dataset, img_right: xr.Dataset, cost_volume: xr.Dataset, disp_min: Union[int, np.ndarray], disp_max: Union[int, np.ndarray]) -> None

      Masks the cost volume :
          - costs which are not inside their disparity range, are masked with a nan value
          - costs of invalid_pixels (invalidated by the input image mask), are masked with a nan value
          - costs of no_data pixels, are masked with a nan value. If a valid pixel contains a no_data in its
              aggregation window, then the cost of the central pixel is masked with a nan value

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk : 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cost_volume: the cost_volume DataSet with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
      :type cost_volume: xarray.Dataset
      :param disp_min: minimum disparity
      :type disp_min: int or np.ndarray
      :param disp_max: maximum disparity
      :type disp_max: int or np.ndarray
      :param cfg: images configuration containing the mask convention : valid_pixels, no_data
      :type cfg: dict
      :return: None



