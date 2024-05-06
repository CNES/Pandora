.. _filter:

Filtering of the disparity map
==============================

Theoretical basics
------------------

The filtering methods allow to homogenize the disparity maps, those available in pandora are :

- median filter.
- bilateral filter.
- median filter for intervals. See :ref:`cost_volume_confidence` for more details.

.. note::  Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is ignored for the calculation


.. list-table:: Configuration and parameters
   :widths: 19 19 19 19 19 19
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - *filter_method*
     - Filtering method
     - str
     -
     - | "median",
       | "bilateral",
       | "median_for_intervals"
     - Yes
   * - *filter_size*
     - Filter's size
     - int
     - 3
     - >=1
     - No. Only available if "median" or "median_for_intervals" filter
   * - *sigma_color*
     - Bilateral filter parameter
     - float
     - 2.0
     - 
     - No. Only available if "bilateral" filter
   * - *sigma_space*
     - Bilateral filter parameter
     - float
     - 6.0
     - 
     - No. Only available if "bilateral" filter
   * - *interval_indicator*
     - | Indicator for which interval to filter.
       | Ex: If *cfg* contains a step "cost_volume_confidence.intervals"
       | then *interval_indicator* should be "intervals"
     - str
     - ""
     - 
     - No. Only available if "median_for_intervals" filter
   * - *regularization*
     - Activate regularization
     - bool
     - false
     - true, false
     - No. Only available if "median_for_intervals" filter
   * - *ambiguity_indicator*
     - | Indicator for which ambiguity to use during regularization.
       | Ex: If *cfg* contains a step "cost_volume_confidence.amb"
       | then *ambiguity_indicator* should be "amb"
     - str
     - ""
     - 
     - No. Only available if "median_for_intervals" filter
   * - *ambiguity_threshold*
     - A pixel is regularized if threshold>ambiguity
     - float
     - 0.6
     - >0 and <1
     - No. Only available if "median_for_intervals" filter
   * - *ambiguity_kernel_size*
     - Ambiguity kernel size for regularization. See publication for details.
     - int
     - 5
     - >=0
     - No. Only available if "median_for_intervals" filter
   * - *vertical_depth*
     - Depth for graph regularization. See publication for details.
     - int
     - 2
     - >=0
     - No. Only available if "median_for_intervals" filter
   * - *quantile_regularization*
     - Quantile used for regularization
     - float
     - 0.9
     - >=0 and <=1
     - No. Only available if "median_for_intervals" filter


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
                "regularization": false
            },
            // ...
            "filter":
            {
                "filter_method": "median"
            },
            "filter.int":
            {
                "filter_method": "median_for_intervals",
                "interval_indicator": "int",  // Filtering intervals computed in 'cost_volume_confidence.int'
                "regularization": true,
                "ambiguity_indicator": "amb"  // Using the ambiguity computed above for regularization
            }
            // ...
        }
    }
