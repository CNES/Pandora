.. _cost_volume_confidence:

Cost volume confidence
======================

Theoretical basics
------------------

The purpose of this step is to compute confidence measure on the cost volume.

The available methods are :

- Ambiguity. This metric is related to a cost curve property and aims to qualify whether a point is ambiguous or not.
  The ambiguity is computed by the following formula :

    .. math::

       Amb(x,y,\eta) &= Card(d \in [d_min,d_max] | cv(x,y,d) < \min_{d}(cv(x,y,d)) +\eta \})

  , where :math:`cv(x,y,d)` is the cost value at pixel :math:`(x,y)` for disparity :math:`d` in disparity range :math:`[d_{min},d_{max}]`.
  From equation ambiguity integral measure is derived and it is defined as the area under the ambiguity curve. Then, ambiguity integral measure
  is converted into a confidence measure :math:`Confidence Ambiguity(x,y) = - Ambiguity Integral`.


- Std images intensity : this metric computes the standard deviation of the intensity.


Configuration and parameters
----------------------------

+--------------------------+-----------------------------------------------+--------+---------------+--------------------------------+------------------------------------------+
| Name                     | Description                                   | Type   | Default value | Available value                | Required                                 |
+==========================+===============================================+========+===============+================================+==========================================+
| *cost_volume_confidence* | Cost volume confidence method                 | string |               | "std_intensity" , "ambiguity"  | Yes                                      |
+--------------------------+-----------------------------------------------+--------+---------------+--------------------------------+------------------------------------------+
| *eta_max*                | Maximum :math:`\eta`                          | float  | 0.7           | >0                             | No. Only available if "ambiguity" method |
+--------------------------+-----------------------------------------------+--------+---------------+--------------------------------+------------------------------------------+
| *eta_step*               | :math:`\eta` step                             | float  | 0.01          | >0                             | No. Only available if "ambiguity" method |
+--------------------------+-----------------------------------------------+--------+---------------+--------------------------------+------------------------------------------+


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
            "cost_volume_confidence":
            {
                "cost_volume_confidence": "ambiguity",
                "eta_max": 0.7,
                "eta_step": 0.01
            }
            ...
        }
    }
