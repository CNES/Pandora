.. _cost_aggregation:

Cost Aggregation
================

Theoretical basics
------------------

The second step is to aggregate the matching costs:

- Cross-based Cost Aggregation [Zhang2009]_. This method consists in creating aggregation support regions that adapt to the structures
  present in the scene, it is performed in 5 steps:

    - a 3x3 median filter is applied to the left image (left image) and the right image (right image),
    - cross support region computation of each pixel of the left image,
    - cross support region computation of each pixel of the right image,
    - combination of the left and right support region,
    - the matching cost is averaged over the combined support region.

.. [Zhang2009] Zhang, K., Lu, J., & Lafruit, G. (2009). Cross-based local stereo matching using orthogonal integral images.
       IEEE transactions on circuits and systems for video technology, 19(7), 1073-1079.

Configuration and parameters
----------------------------

+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| Name                 | Description                                   | Type   | Default value | Available value | Required                            |
+======================+===============================================+========+===============+=================+=====================================+
| *aggregation_method* | Aggregation method                            | string |               | "cbca"          | Yes                                 |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| *cbca_intensity*     | Maximum intensity difference between 2 points | float  | 30.0          | >0              | No. Only available if "cbca" method |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| *cbca_distance*      | Maximum distance difference between 2 points  | int    | 5             | >0              | No. Only available if "cbca" method |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+

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
            "aggregation":
            {
                "aggregation_method": "cbca",
                "cbca_intensity": 25.0
            }
            ...
        }
    }