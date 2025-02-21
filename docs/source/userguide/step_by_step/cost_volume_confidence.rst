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

  The ambiguity integral is by default normalized. However, the user may choose not to perform this normalization by setting the `normalization` parameter to `false`.

`Sarrazin, E., Cournet, M., Dumas, L., Defonte, V., Fardet, Q., Steux, Y., Jimenez Diaz, N., Dubois, E., Youssefi, D., Buffe, F., 2021. Ambiguity concept in stereo matching pipeline.
ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. <https://isprs-archives.copernicus.org/articles/XLIII-B2-2021/383/2021/>`_

.. note ::

 If the user uses pandora while tiling his data, we recommend integrating the add_global_disparity function into the image dataset to avoid a tiling effect. In this case, the ambiguity
 is normalized using a conventional normalization method.

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

- Confidence intervals : This metric computes upper and lower bounds for each point in the disparity map, using possibility distributions :math:`\pi` :

    .. math::
    
        I(i,j) = [\min(D_{i,j}), max(D_{i,j})]
    
    .. math::

        D = \{d~|~\pi_{i,j}(d)\geq threshold\}
    
    .. math::
    
        \pi_{i,j}(d) = 1 - \frac{cv(i,j,d) - \min_\delta cv(i,j,\delta)}{\max_{i,j,\delta}cv(i,j,\delta) - \min_{i,j,\delta}cv(i,j,\delta)}
    
    The threshold used on :math:`\pi` is 0.9 by default, but can be changed by the users.
    
    The intervals sould be regularized in low confidence areas. This can be done directly after computing the intervals, but it is more efficient if done after filtering. To do so, another filtering step using *median_for_intervals* should be added to the pipeline (see :ref:`filter` for more details).

    More info can be found in `Roman Malinowski, Emmanuelle Sarrazin, Loïc Dumas, Emmanuel Dubois, Sébastien Destercke, 2024. Robust Confidence Intervals in Stereo Matching using Possibility Theory - arXiv:2404.06273 [cs]. <https://arxiv.org/abs/2404.06273>`_


.. list-table:: Configuration and parameters
   :widths: 19 19 19 19 19 19
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - *confidence_method*
     - Cost volume confidence method
     - str
     -
     - | "std_intensity",
       | "ambiguity",
       | "risk",
       | "interval_bounds"
     - Yes
   * - *eta_max*
     - Maximum :math:`\eta`
     - float
     - 0.7
     - >0
     - No. Only available if "ambiguity" or "risk" method
   * - *eta_step*
     - :math:`\eta` step
     - float
     - 0.01
     - >0
     - No. Only available if "ambiguity" or "risk" method
   * - *normalization*
     - Ambiguity normalization
     - bool
     - true
     - true, false
     - No. Only available if "ambiguity" method
   * - *possibility_threshold*
     - Threshold on possibility distribution
     - float
     - 0.9
     - >=0 and <=1
     - No. Only available if "interval_bounds" method
   * - *regularization*
     - Activate regularization
     - bool
     - false
     - true, false
     - No. Only available if "interval_bounds" method
   * - *ambiguity_indicator*
     - | Indicator for which ambiguity to use during regularization.
       | Ex: If *cfg* contains a step "cost_volume_confidence.amb"
       | then *ambiguity_indicator* should be "amb"
     - str
     - ""
     - 
     - No. Only available if "interval_bounds" method
   * - *ambiguity_threshold*
     - A pixel is regularized if threshold>ambiguity
     - float
     - 0.6
     - >0 and <1
     - No. Only available if "interval_bounds" method
   * - *ambiguity_kernel_size*
     - Ambiguity kernel size for regularization. See publication for details.
     - int
     - 5
     - >=0
     - No. Only available if "interval_bounds" method
   * - *vertical_depth*
     - Depth for graph regularization. See publication for details.
     - int
     - 2
     - >=0
     - No. Only available if "interval_bounds" method
   * - *quantile_regularization*
     - Quantile used for regularization
     - float
     - 0.9
     - >=0 and <=1
     - No. Only available if "interval_bounds" method


**Example**

.. sourcecode:: json

    {
        "input" :
        {
            // ...
        },
        "pipeline" :
        {
            // ...
            "cost_volume_confidence.amb":
            {
                "confidence_method": "ambiguity",
                "eta_max": 0.7,
                "eta_step": 0.01
            },
            "cost_volume_confidence.int":
            {
                "confidence_method": "interval_bounds",
                "regularization": true,
                "ambiguity_indicator": "amb"  // Using the ambiguity computed above for regularization
            }
            // ...
        }
    }
