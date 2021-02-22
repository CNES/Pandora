:mod:`pandora.filter.median`
============================

.. py:module:: pandora.filter.median

.. autoapi-nested-parse::

   This module contains functions associated to the median filter used to filter the disparity map.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.filter.median.MedianFilter



.. py:class:: MedianFilter(**cfg: Union[str, int])

   Bases: :class:`pandora.filter.filter.AbstractFilter`

   MedianFilter class allows to perform the filtering step

   .. attribute:: _FILTER_SIZE
      :annotation: = 3

      

   .. method:: check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: filter configuration
      :type cfg: dict
      :return cfg: filter configuration updated
      :rtype: dict


   .. method:: desc(self)

      Describes the filtering method


   .. method:: filter_disparity(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None

      Apply a median filter on valid pixels.
      Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
      ignored for the calculation of the median.

      :param disp: the disparity map dataset with the variables :

              - disparity_map 2D xarray.DataArray (row, col)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
              - validity_mask 2D xarray.DataArray (row, col)
      :type disp: xarray.Dataset
      :param img_left: left Dataset image
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image
      :type img_right: xarray.Dataset
      :param cv: cost volume dataset
      :type cv: xarray.Dataset
      :return: None


   .. method:: median_filter(self, data) -> np.ndarray

      Apply median filter on valid pixels (pixels that are not nan).
      Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
      ignored for the calculation of the median.

      :param data: input data to be filtered
      :type data: 2D np.array (row, col)
      :return: The filtered array
      :rtype: 2D np.array(row, col)



