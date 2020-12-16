.. _outputs:

Outputs
=======

Pandora will store several data in the output folder, the tree structure is defined in the file
pandora/output_tree_design.py.

Saved images
************

- *left_disparity.tif*, *right_disparity.tif* : disparity maps in left and right image geometry.

- *left_validity_mask.tif*, *right_validity_mask.tif* : the :ref:`validity_mask` in left image geometry, and
  right. Note that bits 4, 5, 8 and 9 can only be calculated if a validation step is set.

.. note::
    Right products are only available if a *right_disp_map* parameter is activated and not equal to none.
    Must be the first key of *pipeline*.

+-----------------+---------------------------------------------+--------+---------------+--------------------------------+----------+
| Name            | Description                                 | Type   | Default value | Available value                | Required |
+=================+=============================================+========+===============+================================+==========+
| *method*        | Method to compute the right disparity map   | string |   none        | "none", "accurate"             | Yes      |
+-----------------+---------------------------------------------+--------+---------------+--------------------------------+----------+

Method value meaning:

- *none*: the right disparity map is not calculated.
- *accurate*: the right disparity map is calculated following the same pipeline as for the left disparity map, by inverting input images. The left one becomes the right one, the right one becomes the left one.

Example
*******

.. sourcecode:: text

    {
      "input" : {
            ...
      },
      "pipeline" :
       {
            "right_disp_map":
            {
                "method": "accurate",
            }
            ...
        }
    }

.. _validity_mask:

Validity mask
*************

Validity masks indicate why a pixel in the image is invalid and
provide information on the reliability of the match. These masks are 16-bit encoded: each bit
represents a rejection / information criterion (= 1 if rejection / information, = 0 otherwise):

 +---------+--------------------------------------------------------------------------------------------------------+
 | **Bit** | **Description**                                                                                        |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    0    |   * border of left image                                                                               |
 |         |   * nodata of left image                                                                               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    1    |   - Disparity range does not permit to find any point on the right image                               |
 |         |   - nodata of right image                                                                              |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    2    | Information : disparity range cannot be used completely , reaching border of right image               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    3    | Information: calculations stopped at the pixel stage, sub-pixel interpolation was not successful       |
 |         | (for vfit: pixels d-1 and/or d+1 could not be calculated)                                              |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    4    | Information : filled occlusion                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    5    | Information : filled mismatch                                                                          |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    6    | The point is invalid: invalidated by the validity mask associated to the left image                    |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    7    | The point is invalid: right positions to be scanned invalidated by the mask of the right image         |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    8    | The Point is invalid: point located in an occlusion area                                               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    9    | The point is invalid: mismatch                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+
