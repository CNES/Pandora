:mod:`pandora.img_tools`
========================

.. py:module:: pandora.img_tools

.. autoapi-nested-parse::

   This module contains functions associated to raster images.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pandora.img_tools.rasterio_open
   pandora.img_tools.read_img
   pandora.img_tools.check_dataset
   pandora.img_tools.prepare_pyramid
   pandora.img_tools.fill_nodata_image
   pandora.img_tools.interpolate_nodata_sgm
   pandora.img_tools.masks_pyramid
   pandora.img_tools.convert_pyramid_to_dataset
   pandora.img_tools.shift_right_img
   pandora.img_tools.check_inside_image
   pandora.img_tools.census_transform
   pandora.img_tools.compute_mean_raster
   pandora.img_tools.find_valid_neighbors
   pandora.img_tools.compute_mean_patch
   pandora.img_tools.compute_std_raster
   pandora.img_tools.read_disp


.. function:: rasterio_open(*args: str, **kwargs: Union[int, str, None]) -> rasterio.io.DatasetReader

   rasterio.open wrapper to silence UserWarning like NotGeoreferencedWarning.

   (see https://rasterio.readthedocs.io/en/latest/api/rasterio.errors.html)

   :param args: args to be given to rasterio.open method
   :type args: str
   :param kwargs: kwargs to be given to rasterio.open method
   :type kwargs: Union[int, str, None]
   :return: rasterio DatasetReader
   :rtype: rasterio.io.DatasetReader


.. function:: read_img(img: str, no_data: float, mask: str = None, classif: str = None, segm: str = None) -> xr.Dataset

   Read image and mask, and return the corresponding xarray.DataSet

   :param img: Path to the image
   :type img: string
   :type no_data: no_data value in the image
   :type no_data: float
   :param mask: Path to the mask (optional): 0 value for valid pixels, !=0 value for invalid pixels
   :type mask: string
   :param classif: Path to the classif (optional)
   :type classif: string
   :param segm: Path to the mask (optional)
   :type segm: string
   :return: xarray.DataSet containing the variables :

           - im : 2D (row, col) xarray.DataArray float32
           - msk : 2D (row, col) xarray.DataArray int16, with the convention defined in the configuration file
   :rtype: xarray.DataSet


.. function:: check_dataset(dataset: xr.Dataset) -> xr.Dataset

   Check if input dataset is correct, and return the corresponding xarray.DataSet

   :param dataset: dataset
   :type dataset: xr.Dataset
   :return: full dataset
   :rtype: xarray.DataSet


.. function:: prepare_pyramid(img_left: xr.Dataset, img_right: xr.Dataset, num_scales: int, scale_factor: int) -> Tuple[List[xr.Dataset], List[xr.Dataset]]

   Return a List with the datasets at the different scales

   :param img_left: left Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
   :type img_left: xarray.Dataset
   :param img_right: right Dataset containing :

           - im : 2D (row, col) xarray.DataArray
   :type img_right: xarray.Dataset
   :param num_scales: number of scales
   :type num_scales: int
   :param scale_factor: factor by which downsample the images
   :type scale_factor: int
   :return: a List that contains the different scaled datasets
   :rtype: List of xarray.Dataset


.. function:: fill_nodata_image(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]

   Interpolate no data values in image. If no mask was given, create all valid masks

   :param dataset: Dataset image
   :type dataset: xarray.Dataset containing :

       - im : 2D (row, col) xarray.DataArray
   :return: a Tuple that contains the filled image and mask
   :rtype: Tuple of np.ndarray


.. function:: interpolate_nodata_sgm(img: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]

   Interpolation of the input image to resolve invalid (nodata) pixels.
   Interpolate invalid pixels by finding the nearest correct pixels in 8 different directions
   and use the median of their disparities.

   HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
   IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

   :param img: input image
   :type img: 2D np.array (row, col)
   :param valid: validity mask
   :type valid: 2D np.array (row, col)
   :return: the interpolate input image, with the validity mask update :

       - If out & PANDORA_MSK_PIXEL_FILLED_NODATA != 0 : Invalid pixel : filled nodata pixel
   :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))


.. function:: masks_pyramid(msk: np.ndarray, scale_factor: int, num_scales: int) -> List[np.ndarray]

   Return a List with the downsampled masks for each scale

   :param msk: full resolution mask
   :type msk: np.ndarray
   :param scale_factor: scale factor
   :type scale_factor: int
   :param num_scales: number of scales
   :type num_scales: int
   :return: a List that contains the different scaled masks
   :rtype: List of np.ndarray


.. function:: convert_pyramid_to_dataset(img_orig: xr.Dataset, images: List[np.ndarray], masks: List[np.ndarray]) -> List[xr.Dataset]

   Return a List with the datasets at the different scales

   :param img_left: left Dataset image containing :

       - im : 2D (row, col) xarray.DataArray
   :type img_left: xarray.Dataset
   :param img_right: right Dataset image containing :

       - im : 2D (row, col) xarray.DataArray
   :type img_right: xarray.Dataset
   :param num_scales: number of scales
   :type num_scales: int
   :param scale_factor: factor by which downsample the images
   :type scale_factor: int
   :return: a List that contains the different scaled datasets
   :rtype: List of xarray.Dataset


.. function:: shift_right_img(img_right: xr.Dataset, subpix: int) -> List[xr.Dataset]

   Return an array that contains the shifted right images

   :param img_right: right Dataset image containing :

       - im : 2D (row, col) xarray.DataArray
   :type img_right: xarray.Dataset
   :param subpix: subpixel precision = (1 or 2 or 4)
   :type subpix: int
   :return: an array that contains the shifted right images
   :rtype: array of xarray.Dataset


.. function:: check_inside_image(img: xr.Dataset, row: int, col: int) -> bool

   Check if the coordinates row,col are inside the image

   :param img: Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
   :type img: xarray.Dataset
   :param row: row coordinates
   :type row: int
   :param col: column coordinates
   :type col: int
   :return: a boolean
   :rtype: boolean


.. function:: census_transform(image: xr.Dataset, window_size: int) -> xr.Dataset

   Generates the census transformed image from an image

   :param image: Dataset image containing the image im : 2D (row, col) xarray.Dataset
   :type image: xarray.Dataset
   :param window_size: Census window size
   :type window_size: int
   :return: Dataset census transformed uint32 containing the transformed image im: 2D (row, col) xarray.DataArray     uint32
   :rtype: xarray.Dataset


.. function:: compute_mean_raster(img: xr.Dataset, win_size: int) -> np.ndarray

   Compute the mean within a sliding window for the whole image

   :param img: Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
   :type img: xarray.Dataset
   :param win_size: window size
   :type win_size: int
   :return: mean raster
   :rtype: numpy array


.. function:: find_valid_neighbors(dirs: np.ndarray, disp: np.ndarray, valid: np.ndarray, row: int, col: int)

   Find valid neighbors along directions

   :param dirs: directions
   :type dirs: 2D np.array (row, col)
   :param disp: disparity map
   :type disp: 2D np.array (row, col)
   :param valid: validity mask
   :type valid: 2D np.array (row, col)
   :param row: row current value
   :type row: int
   :param col: col current value
   :type col: int
   :return: valid neighbors
   :rtype: 2D np.array


.. function:: compute_mean_patch(img: xr.Dataset, row: int, col: int, win_size: int) -> np.ndarray

   Compute the mean within a window centered at position row,col

   :param img: Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
   :type img: xarray.Dataset
   :param row: row coordinates
   :type row: int
   :param col: column coordinates
   :type col: int
   :param win_size: window size
   :type win_size: int
   :return: mean
   :rtype: float


.. function:: compute_std_raster(img: xr.Dataset, win_size: int) -> np.ndarray

   Compute the standard deviation within a sliding window for the whole image
   with the formula : std = sqrt( E[row^2] - E[row]^2 )

   :param img: Dataset image containing :

           - im : 2D (row, col) xarray.DataArray
   :type img: xarray.Dataset
   :param win_size: window size
   :type win_size: int
   :return: std raster
   :rtype: numpy array


.. function:: read_disp(disparity: Union[None, int, str]) -> Union[None, int, np.ndarray]

   Read the disparity :
       - if cfg_disp is the path of a disparity grid, read and return the grid (type numpy array)
       - else return the value of cfg_disp

   :param disparity: disparity, or path to the disparity grid
   :type disparity: None, int or str
   :return: the disparity
   :rtype: int or np.ndarray


