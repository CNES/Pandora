:mod:`pandora.common`
=====================

.. py:module:: pandora.common

.. autoapi-nested-parse::

   This module contains functions allowing to save the results and the configuration of Pandora pipeline.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pandora.common.write_data_array
   pandora.common.mkdir_p
   pandora.common.save_results
   pandora.common.sliding_window
   pandora.common.save_config
   pandora.common.is_method


.. function:: write_data_array(data_array: xr.DataArray, filename: str, dtype: rasterio.dtypes = rasterio.dtypes.float32, crs: Union[rasterio.crs.CRS, None] = None, transform: rasterio.Affine = rasterio.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)) -> None

   Write a xarray.DataArray in a tiff file

   :param data_array: data
   :type data_array: 2D xarray.DataArray (row, col) or 3D xarray.DataArray (row, col, indicator)
   :param filename:  output filename
   :type filename: string
   :param dtype: band types
   :type dtype: rasterio.dtypes
   :param crs: coordinate reference support
   :type dtype: rasterio.crs.CRS
   :param transform: geospatial transform matrix
   :type dtype: rasterio.Affine
   :return: None


.. function:: mkdir_p(path: str) -> None

   Create a directory without complaining if it already exists.
   :return: None


.. function:: save_results(left: xr.Dataset, right: xr.Dataset, output: str) -> None

   Save results in the output directory

   :param left: left dataset, which contains the variables :

       - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
       - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray         (row, col, indicator)
       - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)
   :type left: xr.Dataset
   :param right: right dataset. If there is no validation step, the right Dataset will be empty.If a validation step     is configured, the dataset will contain the variables :

       - disparity_map: the disparity map in the geometry of the right image 2D DataArray (row, col)
       - confidence_measure: the confidence in the geometry of the right image 3D DataArray (row, col, indicator)
       - validity_mask: the validity mask in the geometry of the left image 2D DataArray (row, col)
   :type right: xr.Dataset
   :param output: output directory
   :type output: string
   :return: None


.. function:: sliding_window(base_array: np.array, shape: Tuple[int, int]) -> np.array

   Create a sliding window of using as_strided function : this function create a new a view (by manipulating
   data pointer) of the data array with a different shape. The new view pointing to the same memory block as
   data so it does not consume any additional memory.

   :param base_array: the 2D array through which slide the window
   :type base_array: np.array
   :param shape: shape of the sliding window
   :type shape: Tuple[int,int]

   :rtype: np.array


.. function:: save_config(output: str, user_cfg: Dict) -> None

   Save the user configuration in json file

   :param output: Path to output directory
   :type output: string
   :param user_cfg: user configuration
   :type user_cfg: dict
   :return: None


.. function:: is_method(string_method: str, methods: List[str]) -> bool

   Test if string_method is a method in methods

   :param string_method: String to test
   :type string_method: string
   :param methods: list of available methods
   :type methods: list of strings
   :returns: True if string_method a method and False otherwise
   :rtype: bool


