Usage
=====

The use as a binary is as follows:

.. sourcecode:: text

    $ pandora config.json output_dir

Required arguments are :

- *config.json*: configuration file. Paramaters are described in :ref:`config_file`
- *output_dir*: le chemin vers le dossier de sortie

The other optional arguments, also available, are :

-
.. note::
    The disparity calculated by Pandora is such that :

        :math:`I_{L}(x, y) = I_{R}(x + d, y)`

    with :math:`I_{L}` , :math:`I_{R}` the reference image (left image) and the secondary image (right image), and
    :math:`d` the disparity.

.. _config_file:

Configuration file
------------------
The configuration file provides a list of parameters to Pandora so that the processing pipeline can
run according to the parameters chosen by the user.

Pandora works with JSON formatted data with the following nested structures.


.. sourcecode:: text

    {
      "input" : {
        ...
      },
      "stereo" : {
        ...
      },
      "aggregation" : {
        ...
      },
      "optimization" : {
        ...
      },
      "refinement": {
        ...
      },
      "filter" : {
       ...
      },
      "validation" : {
        ...
      },
      "invalid_disparity": ...
    }


+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| Name                | Description                       | Type | Default value | Sub structures            | Required |
+=====================+===================================+======+===============+===========================+==========+
| *input*             | Input data to process             | dict |               | :ref:`input_parameters`   | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *image*             | Images and masks parameters       | dict |               | :ref:`image_parameters`   | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *stereo*            | Pixel and mask parameters         | dict |               | :ref:`stereo_parameters`  | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *aggregation*       | Aggregation step parameters       | dict |               | :ref:`aggreg_parameters`  | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *optimization*      | Optimization step parameters      | dict |               | :ref:`optim_parameters`   | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *refinement*        | Refinement step parameters        | dict |               | :ref:`refine_parameters`  | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *filter*            | Filtering step parameters         | dict |               | :ref:`filter_parameters`  | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *validation*        | Validation step parameters        | dict |               | :ref:`valid_parameters`   | Yes      |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+
| *invalid disparity* | Disparity value for invalid pixel | int  | -99999        |                           | No       |
+---------------------+-----------------------------------+------+---------------+---------------------------+----------+

.. _input_parameters:

Input parameters
^^^^^^^^^^^^^^^^

+------------+--------------------------------+--------+---------------+--------------------------+
| Name       | Description                    | Type   | Default value | Required                 |
+============+================================+========+===============+==========================+
|*img_ref*   | Path to the reference image    | string |               | Yes                      |
+------------+--------------------------------+--------+---------------+--------------------------+
| *img_sec*  | Path to the secondary image    | string |               | Yes                      |
+------------+--------------------------------+--------+---------------+--------------------------+
| *disp_min* | minimal disparity              | int    |               | Yes                      |
+------------+--------------------------------+--------+---------------+--------------------------+
| *disp_max* | maximal disparity              | int    |               | Yes                      |
+------------+--------------------------------+--------+---------------+--------------------------+
| *ref_mask* | path to the reference mask     | string | "none"        | No                       |
+------------+--------------------------------+--------+---------------+--------------------------+
| *sec_mask* | path to the secondary mask     | string | "none"        | No                       |
+------------+--------------------------------+--------+---------------+--------------------------+

.. _image_parameters:

Image parameters
^^^^^^^^^^^^^^^^

+--------------+----------------------------------+------+---------------+----------+
| Name         | Description                      | Type | Default value | Required |
+==============+==================================+======+===============+==========+
| nodata1      | Nodata value for reference image | int  | 0             | No       |
+--------------+----------------------------------+------+---------------+----------+
| nodata2      | Nodata value for secondary image | int  | 0             | No       |
+--------------+----------------------------------+------+---------------+----------+
| valid_pixels | Valid pixel value in the mask    | int  | 0             | No       |
+--------------+----------------------------------+------+---------------+----------+
| no_data      | Nodata pixel value in the mask   | int  | 1             | No       |
+--------------+----------------------------------+------+---------------+----------+


.. _stereo_parameters:

Stereo parameters
^^^^^^^^^^^^^^^^^
+-----------------+------------------------------------+--------+---------------+--------------------------------+----------+
| Name            | Description                        | Type   | Default value | Available value                | Required |
+=================+====================================+========+===============+================================+==========+
| *stereo_method* | Similarity measure                 | string | "ssd"         | "ssd" , "sad", "census, "zncc" | Yes      |
+-----------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *window_size*   | Window size for similarity measure | int    | 5             | Must be >0                     | No       |
|                 |                                    |        |               |                                |          |
|                 |                                    |        |               | For "census" : {3,5}           |          |
+-----------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *subpix*        | Cost volume upsampling factor      | int    | 1             | {1,2,4}                        | No       |
+-----------------+------------------------------------+--------+---------------+--------------------------------+----------+

.. note::
    Example for *subpix* parameter with disp_min = 0 and disp_max = 2
        - if *subpix* = 1, cost volume contains {0,1,2} disparities
        - if *subpix* = 2, cost volume contains {0., 0.5, 1., 1.5, 2.} disparities
        - if *subpix* = 4, cost volume containes {0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.} disparities

.. _aggreg_parameters:

Aggregation parameters
^^^^^^^^^^^^^^^^^^^^^^

+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| Name                 | Description                                   | Type   | Default value | Available value | Required                            |
+======================+===============================================+========+===============+=================+=====================================+
| *aggregation_method* | Aggregation method                            | string | "none"        | "cbca"          | Yes                                 |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| *cbca_intensity*     | Maximum intensity difference between 2 points | float  | 30.0          | >0              | No. Only available if "cbca" method |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+
| *cbca_distance*      | Maximum distance difference between 2 points  | int    | 5             | >0              | No. Only available if "cbca" method |
+----------------------+-----------------------------------------------+--------+---------------+-----------------+-------------------------------------+

.. _optim_parameters:

Optimization parameters
^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+----------------------+--------+---------------+-------------------------------------+----------+
| Name                  | Description          | Type   | Default value | Available value                     | Required |
+=======================+======================+========+===============+=====================================+==========+
| *optimization_method* | Optimization méthode | string | "none"        | "sgm" if plugin_libsgm is installed | Yes      |
+-----------------------+----------------------+--------+---------------+-------------------------------------+----------+

.. note:: If plugin_libsgm is installed, see the documentation of this package. There are subparameters for sgm method.

.. _refine_parameters:

Refinement parameters
^^^^^^^^^^^^^^^^^^^^^

+---------------------+-------------------+--------+---------------+---------------------+----------+
| Name                | Description       | Type   | Default value | Available value     | Required |
+=====================+===================+========+===============+=====================+==========+
| *refinement_method* | Refinement method | string | "none"        | "vift", "quadratic" | Yes      |
+---------------------+-------------------+--------+---------------+---------------------+----------+

.. _filter_parameters:

Filtering parameters
^^^^^^^^^^^^^^^^^^^^

+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| Name            | Description                | Type   | Default value | Available value       | Required                           |
+=================+============================+========+===============+=======================+====================================+
| *filter_method* | Filtering method           | string | "none"        | "median", "bilateral" | Yes                                |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *filter_size*   | Filter's size              | int    | 3             | >= 1                  | No                                 |
|                 |                            |        |               |                       | Only avalaible if median filter    |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *sigma_color*   | Bilateral filter parameter | float  | 2.0           |                       | No                                 |
|                 |                            |        |               |                       | Only avalaible if bilateral filter |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *sigma_space*   | Bilateral filter parameter | float  | 6.0           |                       | No                                 |
|                 |                            |        |               |                       |                                    |
|                 |                            |        |               |                       | Only avalaible if bilateral filter |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+

.. _valid_parameters:

Validation parameters
^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| Name                              | Description                                                                                             | Type   | Default value | Available value           | Required |
+===================================+=========================================================================================================+========+===============+===========================+==========+
| *validation_method*               | Validation method                                                                                       | string | "none"        | "cross_checking"          | Yes      |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| *right_left_mode*                 | Method for right disparity map computation                                                              | string | "accurate"    | "accurate"                | No       |
|                                   | - if "accurate": right disparity map computed from scratch                                              |        |               |                           |          |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| *interpolated_disparity*          | Interpolation method for filling occlusion and mismatches                                               | string | "none"        | "mc_cnn", "sgm"           | No       |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| *filter_interpolated_disparities* | Disparity map filtering. If activated, filtering method is the one defined on  :ref:`filter_parameters` | bool   | True          | False                     | No       |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+


Output
-----------

Pandora will store several data in the output folder, the tree structure is defined in the file
pandora/output_tree_design.py.

Saved images

- *ref_disparity.tif*, *sec_disparity.tif* : disparity maps in reference and secondary image geometry.

- *ref_validity_mask.tif*, *sec_validity_mask.tif* : the :ref:`validity_mask` in reference image geometry, and
  secondary. Note that bits 4, 5, 8 and 9 can only be calculated if a validation step is set.

.. note::
    Secondary products are only available if a validation step is
    configured ( ex: validation_method = cross_checking).

.. _validity_mask:

Validity mask
^^^^^^^^^^^^^

Validity masks indicate why a pixel in the image is invalid and
provide information on the reliability of the match. These masks are 16-bit encoded: each bit
represents a rejection / information criterion (= 1 if rejection / information, = 0 otherwise):

 +---------+--------------------------------------------------------------------------------------------------------+
 | **Bit** | **Description**                                                                                        |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    0    |   * border of reference image                                                                          |
 |         |   * nodata of reference image                                                                          |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    1    |   - Disparity range does not permit to find any point on the secondary image                           |
 |         |   - nodata of secondary image                                                                          |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    2    | Information : disparity range cannot be used completely , reaching border of secondary image           |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    3    | Information: calculations stopped at the pixel stage, sub-pixel interpolation was not successful       |
 |         | (for vfit: pixels d-1 and/or d+1 could not be calculated)                                              |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    4    | Information : closed occlusion                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    5    | Information : closed mismatch                                                                          |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    6    | The point is invalid: invalidated by the validity mask associated to the reference image               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    7    | The point is invalid: secondary positions to be scanned invalidated by the mask of the secondary image |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    8    | The Point is invalid: point located in an occlusion area                                               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    9    | The point is invalid: mismatch                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+
