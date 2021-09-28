.. _disparity_refinement:

Disparity refinement
====================

Theoretical basics
------------------

The purpose of this step is to refine the disparity identified in the previous step. It consists in interpolating the
coefficients of similarity.

.. note::  The cost volume disparity dimension is sampled at the input images rate by default.
           Thus disparities tested are integers. However, to prevent from aliasing effects when
           refining the disparity map, one can use the *subpix* parameter.
           This will add subpixel disparities into the cost volume by oversampling the disparity dimension by an even factor.

The available interpolation methods are :

- Vfit [Haller2010]_, consists in estimating a symmetrical form V from 3 points: the disparity :math:`d` identified at
  the previous step as well as :math:`d - 1` and :math:`d + 1` with their costs. The following figure
  represents the function to be estimated :

    .. image:: ../../Images/Vfit.png
        :width: 300px
        :height: 200px


  The interpolation is given by the following formula, where :math:`c` the matching cost, and :math:`p` the slope :

    .. math::

       y &= c(d + 1) + (x - 1) * p  \\
       y &= c(d - 1) + (x - (-1)) * -p  \\
       x &= (c(d - 1) - c(d + 1)) / (2*p)

- Quadratic, consists in estimating a parabola from 3 points: the disparity :math:`d` identified at
  the previous step as well as :math:`d - 1` and :math:`d + 1` with their costs. The following figure
  represents the function to be estimated :

    .. image:: ../../Images/Quadratic.png
        :width: 300px
        :height: 200px

    .. math::

       y &= ax^2 + bx + c \\
       a &= (c(d-1) - 2*c(d) + c(d+1) / 2 \\
       b &= (c(d+1) - c(d-1)) / 2 \\
       c &= c(d) \\
       x &= -b / 2a \\


.. [Haller2010] HALLER, István, PANTILIE, C., ONIGA, F., et al. Real-time semi-global dense stereo solution with improved
       sub-pixel accuracy. In : 2010 IEEE Intelligent Vehicles Symposium. IEEE, 2010. p. 369-376.

.. note:: If one of the coefficients is invalid, :math:`d`, :math:`d - 1` or :math:`d + 1`, interpolation is not performed.


Configuration and parameters
----------------------------

+---------------------+-------------------+--------+---------------+---------------------+----------+
| Name                | Description       | Type   | Default value | Available value     | Required |
+=====================+===================+========+===============+=====================+==========+
| *refinement_method* | Refinement method | string |               | "vfit", "quadratic" | Yes      |
+---------------------+-------------------+--------+---------------+---------------------+----------+

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
            "refinement":
            {
               "refinement_method": "vfit"
            }
            ...
        }
    }

