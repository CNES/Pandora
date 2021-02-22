:mod:`pandora.cost_volume_confidence.cost_volume_confidence`
============================================================

.. py:module:: pandora.cost_volume_confidence.cost_volume_confidence

.. autoapi-nested-parse::

   This module contains classes and functions to estimate confidence.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.cost_volume_confidence.cost_volume_confidence.AbstractCostVolumeConfidence



.. py:class:: AbstractCostVolumeConfidence

   Abstract Cost Volume Confidence class

   .. attribute:: __metaclass__
      

      

   .. attribute:: confidence_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self)
      :abstractmethod:

      Describes the confidence method


   .. method:: confidence_prediction(self, disp: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset) -> None
      :abstractmethod:

      Computes a confidence prediction.

      :param disp: the disparity map dataset or None
      :type disp: xarray.Dataset or None
      :param img_left: left Dataset image
      :tye img_left: xarray.Dataset
      :param img_right: right Dataset image
      :type img_right: xarray.Dataset
      :param cv: cost volume dataset
      :type cv: xarray.Dataset
      :return: None


   .. method:: allocate_confidence_map(name_confidence_measure: str, confidence_map: np.ndarray, disp: xr.Dataset, cv: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]
      :staticmethod:

      Create or update the confidence measure : confidence_measure (xarray.DataArray of the cost volume and the
      disparity map) by adding a the indicator

      :param name_confidence_measure: the name of the new confidence indicator
      :type name_confidence_measure: string
      :param confidence_map: the condidence map
      :type confidence_map: 2D np.array (row, col) dtype=np.float32
      :param disp: the disparity map dataset or None
      :type disp: xarray.Dataset or None
      :param cv: cost volume dataset
      :type cv: xarray.Dataset
      :return: the disparity map and the cost volume with updated confidence measure
      :rtype:
          Tuple(xarray.Dataset, xarray.Dataset) with the data variables:
              - confidence_measure 3D xarray.DataArray (row, col, indicator)



