.. _cost_volume_confidence:

Cost volume confidence
======================

Theoretical basics
------------------

The purpose of this step is to compute confidence measure on the cost volume.

The available methods are :

- Ambiguity. This metric is related to a cost curve property and aims to qualify whether a point is ambiguous or not.
  From one pixel, the ambiguity is computed by the following formula :

    .. math::

       Amb(x,y,\eta) &= Card(d \in [d_min,d_max] | cv(x,y,d) < \min_{d}(cv(x,y,d)) +\eta \})

  , where :math:`cv(x,y,d)` is the cost value at pixel :math:`(x,y)` for disparity :math:`d` in disparity range :math:`[d_{min},d_{max}]`.
  From the previous equation, ambiguity integral measure is derived and it is defined as the area under the ambiguity curve. Then, ambiguity integral measure
  is converted into a confidence measure :math:`Confidence Ambiguity(x,y) = 1 - Ambiguity Integral`.

*Sarrazin, E., Cournet, M., Dumas, L., Defonte, V., Fardet, Q., Steux, Y., Jimenez Diaz, N., Dubois, E., Youssefi, D., Buffe, F., 2021. Ambiguity concept in stereo matching pipeline.
ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. (To be published)*


- Risk. This metric consists in evaluating a risk interval associated with the correlation measure, and ultimately the selected disparity, for each point on the disparity map :

    .. math::

        Risk(x,y,\eta) &= max(d) - min(d) \text{ for d "within" } [c_{min}(x,y) ; c_{min}(x,y)+k\eta[

    .. math::

        Risk_{min}(x,y) &= \text{mean}_\eta( (1+Risk(x,y,\eta)) - Amb(x,y,\eta))

    .. math::

        Risk_{max}(x,y) &= \text{mean}_\eta( Risk(x,y,\eta))


    , where :math:`c_{min}` is the minimum cost value of :math:`cv(x,y)` at pixel :math:`(x,y)`.
    Disparities for every similarity value outside of :math:`[c_{min}(x,y) ; c_{min}(x,y)+k\eta[[min;min+eta[` are discarded for the risk computation.

    From the previous equations, :math:`risk_{min}(x,y)` measure is defined as the mean of :math:`(1+Risk(x,y,\eta)) - Amb(x,y,\eta)` values over :math:`\eta`, whilst :math:`risk_{max}(x,y)` measure is defined as the mean of :math:`Risk(x,y,k)` over :math:`\eta`.


- Std images intensity : this metric computes the standard deviation of the intensity.




Configuration and parameters
----------------------------

+---------------------------+-----------------------------------------------+--------+---------------+----------------------------------------+----------------------------------------------------+
| Name                      | Description                                   | Type   | Default value | Available value                        | Required                                           |
+===========================+===============================================+========+===============+========================================+====================================================+
| *confidence_method*       | Cost volume confidence method                 | string |               | "std_intensity" , "ambiguity", "risk"  | Yes                                                |
+---------------------------+-----------------------------------------------+--------+---------------+----------------------------------------+----------------------------------------------------+
| *eta_max*                 | Maximum :math:`\eta`                          | float  | 0.7           | >0                                     | No. Only available if "ambiguity" or "risk" method |
+---------------------------+-----------------------------------------------+--------+---------------+----------------------------------------+----------------------------------------------------+
| *eta_step*                | :math:`\eta` step                             | float  | 0.01          | >0                                     | No. Only available if "ambiguity" or "risk" method |
+---------------------------+-----------------------------------------------+--------+---------------+----------------------------------------+----------------------------------------------------+

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
                "confidence_method": "ambiguity",
                "eta_max": 0.7,
                "eta_step": 0.01
            }
            ...
        }
    }
