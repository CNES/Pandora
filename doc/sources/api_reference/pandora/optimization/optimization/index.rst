:mod:`pandora.optimization.optimization`
========================================

.. py:module:: pandora.optimization.optimization

.. autoapi-nested-parse::

   This module contains classes and functions associated to the cost volume optimization step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.optimization.optimization.AbstractOptimization



.. py:class:: AbstractOptimization

   Abstract Optimizationinput class

   .. attribute:: __metaclass__
      

      

   .. attribute:: optimization_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self) -> None
      :abstractmethod:

      Describes the optimization method
      :return: None


   .. method:: optimize_cv(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset
      :abstractmethod:

      Optimizes the cost volume

      :param cv: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :param img_left: left Dataset image
      :type img_left: xarray.DataArray
      :param img_right: right Dataset image
      :type img_right: xarray.DataArray
      :return: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :rtype: xarray.Dataset



