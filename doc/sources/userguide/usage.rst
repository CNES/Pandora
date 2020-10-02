Usage
*****

The use as a binary is as follows:

.. sourcecode:: text

    $ pandora config.json output_dir

Required arguments are :

- *config.json*: configuration file. Paramaters are described in :ref:`config_file`
- *output_dir*: le chemin vers le dossier de sortie

The other optional arguments, also available, are :

- -v for verbose mode
- -h for help

.. note::
    The disparity calculated by Pandora is such that :

        :math:`I_{L}(x, y) = I_{R}(x + d, y)`

    with :math:`I_{L}` , :math:`I_{R}` the reference image (left image) and the secondary image (right image), and
    :math:`d` the disparity.

.. _config_file:

Configuration file
==================
The configuration file provides a list of parameters to Pandora so that the processing pipeline can
run according to the parameters choosen by the user.

Pandora works with JSON formatted data with the following nested structures.


.. sourcecode:: text

    {
      "input" : {
        ...
      },

      "image" : {
        ...
      },

      "pipeline" :
       {
          "stereo" : {
            ...
          },
          "aggregation" : {
            ...
          },
          "optimization" : {
            ...
          },
          "disparity" : {
            ...
          }
          "refinement": {
            ...
          },
          "filter" : {
           ...
          },
          "validation" : {
            ...
          },
          "resize" : {
            ...
          }
      }
    }

+---------------------+-----------------------------------+------+---------------+------------------------------+---------+
| Name                | Description                       | Type | Default value | Sub structures              | Required |
+=====================+===================================+======+===============+=============================+==========+
| *input*             | Input data to process             | dict |               | :ref:`input_parameters`     | Yes      |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *image*             | Images and masks parameters       | dict |               | :ref:`image_parameters`     | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *pipeline*          | Pipeline steps parameters         | dict |               | :ref:`pipeline_parameters`  | Yes      |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+


.. _input_parameters:

Input parameters
----------------

+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| Name           | Description                                               | Type          | Default value | Required |
+================+===========================================================+===============+===============+==========+
|*img_ref*       | Path to the reference image                               | string        |               | Yes      |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *img_sec*      | Path to the secondary image                               | string        |               | Yes      |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *disp_min*     | minimal disparity                                         | int or string |               | Yes      |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *disp_max*     | maximal disparity                                         | int or string |               | Yes      |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *ref_mask*     | path to the reference mask                                | string        | "none"        | No       |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *sec_mask*     | path to the secondary mask                                | string        | "none"        | No       |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *disp_min_sec* | Path to the minimal disparity grid of the secondary image | string        | "none"        | No       |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+
| *disp_max_sec* | Path to the maximal disparity grid of the secondary image | string        | "none"        | No       |
+----------------+-----------------------------------------------------------+---------------+---------------+----------+

.. note::
    - Parameters *disp_min* and *disp_max* can be the disparity range (type int) or the path to the grids
      that contain the minimum and maximum disparity of a pixel (type string).
    - If *disp_min* and *disp_max* are integers, then the range of disparities is fixed. The minimal and maximal
      disparity of the secondary image is automatically calculated : *disp_min_sec* = - *disp_max* and *disp_max_sec* = - *disp_min*.
    - If *disp_min* or *disp_max* are strings, that means they are grids of disparities which have the same size as the input images.
      Each pixel (x,y) of the grid corresponds to a local disparity (min for disp_min and max for disp_max) related to the same pixel (x, y) of the image.
    - Cross-checking step is not applicable if *disp_min*, *disp_max* are path to the reference grids and *disp_min_sec*, *disp_max_sec* are none.


.. _image_parameters:

Image parameters
----------------

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


.. _pipeline_parameters:

Pipeline parameters
-------------------

"Pipeline" parameters define steps sequencing to be run. Pandora will check if sub-parameters of each mentioned step are correct.

+---------------------+-----------------------------------+------+---------------+------------------------------+---------+
| Name                | Description                       | Type | Default value | Sub structures              | Required |
+=====================+===================================+======+===============+=============================+==========+
| *right_disp_map*    | Input data to process             | dict |               | :ref:`rdm_parameters`       | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *stereo*            | Pixel and mask parameters         | dict |               | :ref:`stereo_parameters`    | Yes      |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *aggregation*       | Aggregation step parameters       | dict |               | :ref:`aggreg_parameters`    | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *optimization*      | Optimization step parameters      | dict |               | :ref:`optim_parameters`     | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *disparity *        | Disparity  step parameters        | dict |               | :ref:`disparity_parameters` | Yes      |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *refinement*        | Refinement step parameters        | dict |               | :ref:`refine_parameters`    | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *filter*            | Filtering step parameters         | dict |               | :ref:`filter_parameters`    | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *validation*        | Validation step parameters        | dict |               | :ref:`valid_parameters`     | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+
| *resize*            | Resize step parameters            | dict |               | :ref:`resize_parameters`    | No       |
+---------------------+-----------------------------------+------+---------------+-----------------------------+----------+

.. _rdm_parameters:

Right disparity map parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
+-----------------+---------------------------------------------+--------+---------------+--------------------------------+----------+
| Name            | Description                                 | Type   | Default value | Available value                | Required |
+=================+=============================================+========+===============+================================+==========+
| *stereo_method* | Method to compute the right disparity map   | string |   none        | "none", "accurate"             | Yes      |
+-----------------+---------------------------------------------+--------+---------------+--------------------------------+----------+

.. note::
    * method = "none": the right disparity map is not calculated.
    * method = "accurate": the right disparity map is calculated following the same pipeline as for the left disparity map, by inverting input images:
                           the left one becomes the right one, the right one becomes the left one.


.. _stereo_parameters:

Stereo parameters
^^^^^^^^^^^^^^^^^
+-----------------+------------------------------------+--------+---------------+--------------------------------+----------+
| Name            | Description                        | Type   | Default value | Available value                | Required |
+=================+====================================+========+===============+================================+==========+
| *stereo_method* | Similarity measure                 | string |               | "ssd" , "sad", "census, "zncc" | Yes      |
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
| *aggregation_method* | Aggregation method                            | string |               | "cbca"          | Yes                                 |
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
| *optimization_method* | Optimization method  | string |               | "sgm" if plugin_libsgm is installed | Yes      |
+-----------------------+----------------------+--------+---------------+-------------------------------------+----------+

.. note:: If plugin_libsgm is installed, see the documentation of this package. There are subparameters for sgm method.

.. _disparity_parameters:

Disparity  parameters
^^^^^^^^^^^^^^^^^^^^^

+---------------------+--------------------------+------------+---------------+---------------------+----------+
| Name                | Description              | Type       | Default value | Available value     | Required |
+=====================+==========================+============+===============+=====================+==========+
| *disparity _method* | disparity method         | string     |               | "wta"               | Yes      |
+---------------------+--------------------------+------------+---------------+---------------------+----------+
| *invalid_disparity* | invalid disparity value  | int, float |     -9999     | "np.nan" for NaN    | No       |
+---------------------+--------------------------+------------+---------------+---------------------+----------+

.. _refine_parameters:

Refinement parameters
^^^^^^^^^^^^^^^^^^^^^

+---------------------+-------------------+--------+---------------+---------------------+----------+
| Name                | Description       | Type   | Default value | Available value     | Required |
+=====================+===================+========+===============+=====================+==========+
| *refinement_method* | Refinement method | string |               | "vift", "quadratic" | Yes      |
+---------------------+-------------------+--------+---------------+---------------------+----------+

.. _filter_parameters:

Filtering parameters
^^^^^^^^^^^^^^^^^^^^

+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| Name            | Description                | Type   | Default value | Available value       | Required                           |
+=================+============================+========+===============+=======================+====================================+
| *filter_method* | Filtering method           | string |               | "median", "bilateral" | Yes                                |
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
| *validation_method*               | Validation method                                                                                       | string |               | "cross_checking"          | Yes      |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| *right_left_mode*                 | Method for right disparity map computation                                                              | string | "accurate"    | "accurate"                | No       |
|                                   | - if "accurate": right disparity map computed from scratch                                              |        |               |                           |          |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+
| *interpolated_disparity*          | Interpolation method for filling occlusion and mismatches                                               | string |               | "mc_cnn", "sgm"           | No       |
+-----------------------------------+---------------------------------------------------------------------------------------------------------+--------+---------------+---------------------------+----------+

.. note::
  Cross-checking method cannot be choosen if right disparity map is not calculated. See  :ref:`rdm_parameters` to activate it.

.. _resize_parameters:

Resize  parameters
^^^^^^^^^^^^^^^^^^

+---------------------+--------------------------+------------+---------------+---------------------+----------+
| Name                | Description              | Type       | Default value | Available value     | Required |
+=====================+==========================+============+===============+=====================+==========+
| *border_disparity*  | border  disparity value  | int, float |               | "np.nan" for NaN    | Yes      |
+---------------------+--------------------------+------------+---------------+---------------------+----------+

.. note::
  See :ref:`border_management` to understand the goal of this step.

Sequencing of Pandora steps (Pandora Machine)
---------------------------------------------

Moreover, Pandora will check if the requested steps sequencing is correct following the permitted
transition defined by the Pandora Machine (`transitions <https://github.com/pytransitions/transitions>`_)

Pandora Machine defines 4 possible states:
 - begin
 - cost_volume
 - disparity_map
 - Resized_disparity

It starts at the begin state. To go from a state from another one, transitions are called and triggered
by specific name. It corresponds to the name of Pandora steps you can write in configuration file.

The following diagram highligts all states and possible transitions.

    .. figure:: ../Images/Machine_state_diagram.png

A transition (i.e a pandora's step) can be triggered several times. You must respect the following
naming convention: *stepname_xxx* . xxx can be the string you want.
See :ref:`multiple_filters_example`

Examples
========

SSD measurment and filtered disparity map
-----------------------------------------

Configuration to produce a disparity map, computed by SSD method, and filterd by
median filter method.

.. sourcecode:: text

    {
      "input": {
        "ref_mask": null,
        "sec_mask": null,
        "disp_min_sec": null,
        "disp_max_sec": null,
        "img_ref": "img_ref.png",
        "img_sec": "img_ref.png",
        "disp_min": -100,
        "disp_max": 100
      },
      "pipeline": {
          "stereo": {
            "stereo_method": "ssd",
            "window_size": 5,
            "subpix": 1
          },
          "disparity": {
            "disparity_method": "wta",
            "invalid_disparity": "np.nan"
          },
          "filter": {
            "filter_method": "median"
          }
          "resize": {
            "border_disparity": "np.nan"
          }
      }
    }

An impossible sequencing
------------------------

.. sourcecode:: text

    {
      "input": {
        "ref_mask": null,
        "sec_mask": null,
        "disp_min_sec": null,
        "disp_max_sec": null,
        "img_ref": "img_ref.png",
        "img_sec": "img_ref.png",
        "disp_min": -100,
        "disp_max": 100
      },
      "pipeline": {
          "stereo": {
            "stereo_method": "ssd",
            "window_size": 5,
            "subpix": 1
          },
          "filter": {
            "filter_method": "median"
          }
          "disparity": {
            "disparity_method": "wta",
            "invalid_disparity": "np.nan"
          },
          "filter": {
            "filter_method": "median"
          }
     }
    }

With this configuration, you receive the following error

.. sourcecode:: text

    Problem during Pandora checking configuration steps sequencing. Check your configuration file.
    (...)
    transitions.core.MachineError: "Can't trigger event filter from state cost_volume!"

Before the start, Pandora Machine is in the "begin" state. The configuration file defines *stereo* as
the first step to be triggered. So, Pandora Machine go from *begin* state to *cost_volume* state.
Next, the *filter* is going to be triggered but this is not possible. This step can be triggered only
if the Pandora Machine is in *reference_disparity* or *reference_and_secondary_disparity*.

.. _multiple_filters_example:

Same step, multiple times
-------------------------

.. sourcecode:: text

    {
      "input": {
        "ref_mask": null,
        "sec_mask": null,
        "disp_min_sec": null,
        "disp_max_sec": null,
        "img_ref": "img_ref.png",
        "img_sec": "img_ref.png",
        "disp_min": -100,
        "disp_max": 100
      },
      "pipeline": {
          "stereo": {
            "stereo_method": "ssd",
            "window_size": 5,
            "subpix": 1
          },
          "disparity": {
            "disparity_method": "wta",
            "invalid_disparity": "np.nan"
          },
          "filter_1": {
            "filter_method": "median"
          }
          "filter_2": {
            "filter_method": "bilateral"
          }
     }
    }


Output
======

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
-------------

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
