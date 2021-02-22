:mod:`pandora.validation.validation`
====================================

.. py:module:: pandora.validation.validation

.. autoapi-nested-parse::

   This module contains classes and functions associated to the validation step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.validation.validation.AbstractValidation
   pandora.validation.validation.CrossChecking



.. py:class:: AbstractValidation

   Abstract Validation class

   .. attribute:: __metaclass__
      

      

   .. attribute:: validation_methods_avail
      

      

   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self) -> None
      :abstractmethod:

      Describes the validation method
      :return: None


   .. method:: disparity_checking(self, dataset_left: xr.Dataset, dataset_right: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> xr.Dataset
      :abstractmethod:

      Determination of occlusions and false matches by performing a consistency check on valid pixels.
      Update the validity_mask :

          - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
          - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel

      | Update the measure map: add the disp RL / disp LR distances

      :param dataset_left: left Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type dataset_left: xarray.Dataset
      :param dataset_right: right Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type dataset_right: xarray.Dataset
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
      :return: the left dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col) with the bit 8 and 9 of the validity_mask :
              - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
              - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
      :rtype: xarray.Dataset



.. py:class:: CrossChecking(**cfg)

   Bases: :class:`pandora.validation.validation.AbstractValidation`

   CrossChecking class allows to perform the validation step

   .. attribute:: _THRESHOLD
      :annotation: = 1.0

      

   .. method:: check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: optimization configuration
      :type cfg: dict
      :return: optimization configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the validation method
      :return: None


   .. method:: disparity_checking(self, dataset_left: xr.Dataset, dataset_right: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None, cv: xr.Dataset = None) -> xr.Dataset

      Determination of occlusions and false matches by performing a consistency check on valid pixels.

      Update the validity_mask :

          - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
          - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel

      | Update the measure map: add the disp RL / disp LR distances

      :param dataset_left: left Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - validity_mask 2D xarray.DataArray (row, col)
      :type dataset_left: xarray.Dataset
      :param dataset_right: right Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - validity_mask 2D xarray.DataArray (row, col)
      :type dataset_right: xarray.Dataset
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
      :return: the left dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col) with the bit 8 and 9 of the validity_mask :

              - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
              - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
      :rtype: xarray.Dataset



