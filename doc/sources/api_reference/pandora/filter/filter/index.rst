:mod:`pandora.filter.filter`
============================

.. py:module:: pandora.filter.filter

.. autoapi-nested-parse::

   This module contains classes and functions associated to the disparity map filtering.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.filter.filter.AbstractFilter



.. py:class:: AbstractFilter

   Abstract Filter class

   .. attribute:: __metaclass__
      

      

   .. attribute:: filter_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self)
      :abstractmethod:

      Describes the filtering method


   .. method:: filter_disparity(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> None
      :abstractmethod:

      Post processing the disparity map by applying a filter on valid pixels

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



