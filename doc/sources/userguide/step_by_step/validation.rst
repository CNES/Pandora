.. _validation:

Validation of the disparity map
===============================

Theoretical basics
------------------

Validation methods provide a confidence index on the calculated disparity, those available in pandora are


- The cross checking ( cross checking [Fua1993]_ ), which allows to invalidate disparities. It consists in reversing the role
  of the images (the left image becomes the right image, and vice versa) and to compare the disparity :math:`disp_{L}`
  (corresponding to the left image  :math:`I_{L}` ) with :math:`disp_{R}` (corresponding to the right image :math:`I_{R}` ) :

    - If :math:`| disp_{L}(p) + disp_{R}(p + disp_{L}(p)) | \leq threshold`, then point p is valid
    - If :math:`| disp_{L}(p) + disp_{R}(p + disp_{L}(p)) | \geq threshold`, then point p is invalid

  The threshold is 1 by default, but it can be changed with the *cross_checking_threshold* parameter.
  Pandora will then distinguish between occlusion and mismatch by following the methodology outlined in [Hirschmuller2007]_.
  For each pixel p of the left image invalidated by the cross-checking :

    - If there is a disparity d such as :math:`disp_{R}(p+d)=-d`, it is a mismatch.
    - Otherwise, it's an occlusion.


.. note::  Cross checking does not modify the disparity map, it only informs bits 8 and 9 in the
           validity mask.

It is possible to fill in occlusions and mismatches detected during cross-validation:

- using the method proposed in [Zbontar2016]_ : the disparity of an occluded pixel is modified using the
  first valid disparity from the left. The disparity of a pixel considered as a mismatch becomes the
  median of the first 16 valid pixels in the directions shown below (note: these directions are not related to the libSGM ):


    .. figure:: ../../Images/Directions_mc_cnn.png
        :width: 300px
        :height: 200px

- using the method proposed in [Hirschmuller2007]_ : the disparity of an occluded pixel is modified using the smallest disparity (the disparity closest to 0) in 8 directions.
  The disparity of a pixel considered to be a
  mismatch becomes the median of the first 8 valid pixels in the directions shown below. Mismatches that are direct neighbours of
  occluded pixel are treated as occlusions.

    .. figure:: ../../Images/Directions_interpolation_sgm.png
        :width: 300px
        :height: 200px

.. note::  The parameter *interpolated_disparity* is used to select the method to correct occlusions and mismatches.

.. [Fua1993] Fua, P. (1993). A parallel stereo algorithm that produces dense depth maps and preserves image features.
       Machine vision and applications, 6(1), 35-49.

Configuration and parameters
----------------------------

+-----------------------------------+---------------------------------------------------------------------------------------------------------+------------+---------------+---------------------------+----------+
| Name                              | Description                                                                                             | Type       | Default value | Available value           | Required |
+===================================+=========================================================================================================+============+===============+===========================+==========+
| *validation_method*               | Validation method                                                                                       | string     |               | "cross_checking"          | Yes      |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+------------+---------------+---------------------------+----------+
| *cross_checking_threshold*        | Threshold for cross-checking method                                                                     | int, float | 1.0           |                           | No       |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+------------+---------------+---------------------------+----------+
| *interpolated_disparity*          | Interpolation method for filling occlusion and mismatches                                               | string     |               | "mc_cnn", "sgm"           | No       |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+------------+---------------+---------------------------+----------+

**Example**

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
            "validation":
            {
               "validation_method": "cross_checking"
            }
            ...
        }
    }