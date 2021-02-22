:mod:`pandora.aggregation.aggregation`
======================================

.. py:module:: pandora.aggregation.aggregation

.. autoapi-nested-parse::

   This module contains classes and functions associated to the cost volume aggregation step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.aggregation.aggregation.AbstractAggregation



.. py:class:: AbstractAggregation

   Abstract Aggregation class

   .. attribute:: __metaclass__
      

      

   .. attribute:: aggreg_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self)
      :abstractmethod:

      Describes the aggregation method


   .. method:: cost_volume_aggregation(self, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset, **cfg: Union[str, int]) -> None
      :abstractmethod:

      Aggregate the cost volume for a pair of images

      :param img_left: left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :type img_left: xarray.Dataset
      :param img_right: right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
      :type img_right: xarray.Dataset
      :param cv: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :param cfg: images configuration containing the mask convention : valid_pixels, no_data
      :type cfg: dict
      :return: None



