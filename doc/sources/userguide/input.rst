.. _inputs:

Inputs
======

Pandora works with two stereo rectified one-channel images.


Configuration and parameters
****************************

+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| Name             | Description                                               | Type                  | Default value | Required |
+==================+===========================================================+=======================+===============+==========+
| *img_left*       | Path to the left image                                    | string                |               | Yes      |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *img_right*      | Path to the right image                                   | string                |               | Yes      |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *nodata_left*    | Nodata value for left image                               | int, NaN, inf or -inf | -9999         | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *nodata_right*   | Nodata value for right image                              | int, NaN, inf or -inf | -9999         | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *disp_min*       | Minimal disparity                                         | int or string         |               | Yes      |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *disp_max*       | Maximal disparity                                         | int or string         |               | Yes      |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *left_mask*      | Path to the left mask                                     | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *right_mask*     | Path to the right mask                                    | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *disp_min_right* | Path to the minimal disparity grid of the right image     | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *disp_max_right* | Path to the maximal disparity grid of the right image     | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *left_classif*   | Path to the left classification map                       | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *right_classif*  | Path to the right classification map                      | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *left_segm*      | Path to the left segmentation map                         | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+
| *right_segm*     | Path to the right segmentation map                        | string                | "none"        | No       |
+------------------+-----------------------------------------------------------+-----------------------+---------------+----------+

.. note::
    - Parameters *disp_min* and *disp_max* can be the disparity range (type int) or the path to the grids
      that contain the minimum and maximum disparity of a pixel (type string).
    - If *disp_min* and *disp_max* are integers, then the range of disparities is fixed. The minimal and maximal
      disparity of the right image are automatically calculated : *disp_min_right* = - *disp_max* and *disp_max_right* = - *disp_min*.
    - If *disp_min* or *disp_max* are strings, that means they are paths to grids of disparities which have the same size as the input images.
      Each pixel (x,y) of the grid corresponds to a local disparity (min for disp_min and max for disp_max) related to the same pixel (x, y) of the image.
    - Cross-checking step is not applicable if only left grids are provided (i.e the right one must be provided).

.. note::
    Mask must comply with the following convention
     - Value equal to 0 for valid pixel
     - Value not equal to 0 for invalid pixel

**Example**

.. sourcecode:: text

    {
        "input":
        {
            "img_left": "tests/pandora/left.png",
            "img_right": "tests/pandora/right.png",
            "disp_min": -60,
            "disp_max": 0
        }
        ,
        "pipeline" :
        {
            ...
        }
    }