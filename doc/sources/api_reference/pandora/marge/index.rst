:mod:`pandora.marge`
====================

.. py:module:: pandora.marge

.. autoapi-nested-parse::

   This module contains the function which defines the images margins.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pandora.marge.get_margins


.. function:: get_margins(disp_min: int, disp_max: int, cfg: Dict[str, dict]) -> xr.DataArray

   Calculates the margins for the left and right images according to the configuration

   :param disp_min: minimal disparity
   :type disp_min: int
   :param disp_max: maximal disparity
   :type disp_max: int
   :param cfg: user configuration
   :type cfg: dict of dict
   :return: margin for the images, 2D (image, corner) DataArray, with the dimensions image =      ['left_margin', 'right_margin'], corner = ['left', 'up', 'right', 'down']
   :rtype: DataArray


