.. _disparity:

Disparity computation
=====================

Theoretical basics
------------------

This step looks for the disparity for each pixel of the image that produces the best matching cost:
it's called the Winner Takes All strategy.

The disparity calculated by Pandora is such that:

    :math:`I_{L}(x, y) = I_{R}(x + d, y)`

with :math:`I_{L}` , :math:`I_{R}` the left image (left image) and the right image (right image), and
:math:`d` the disparity.

Configuration and parameters
----------------------------

+---------------------+--------------------------+------------+---------------+---------------------+----------+
| Name                | Description              | Type       | Default value | Available value     | Required |
+=====================+==========================+============+===============+=====================+==========+
| *disparity _method* | Disparity method         | string     |               | "wta"               | Yes      |
+---------------------+--------------------------+------------+---------------+---------------------+----------+
| *invalid_disparity* | Invalid disparity value  | int, float |     -9999     | "NaN" for nan value | No       |
+---------------------+--------------------------+------------+---------------+---------------------+----------+

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
            "disparity":
            {
                "disparity _method": "wta",
                "invalid_disparity": "NaN"
            }
            ...
        }
    }