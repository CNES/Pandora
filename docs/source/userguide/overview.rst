Overviews
=========

Diagram
*******

The following interactive diagram highlights all stereo steps avalaible in Pandora.

.. image:: ../Images/white_box.png
    :align: left

.. image:: ../Images/main_diagram_step_multiscale.png
    :align: right
    :target: step_by_step/multiscale.html

.. image:: ../Images/main_diagram_images_stereo.png
    :align: center
    :target: input.html

.. image:: ../Images/white_box.png
    :align: center

.. image:: ../Images/arrow.png
    :align: center

.. image:: ../Images/main_diagram_step_matching_cost.png
    :align: center
    :target: step_by_step/matching_cost.html

.. image:: ../Images/arrow.png
    :align: center

.. image:: ../Images/main_diagram_step_cost_aggregation.png
    :align: left
    :target: step_by_step/aggregation.html

.. image:: ../Images/main_diagram_step_optimization.png
    :align: right
    :target: step_by_step/optimization.html

.. image:: ../Images/main_diagram_step_cost_volume_confidence.png
    :align: center
    :target: step_by_step/cost_volume_confidence.html

.. image:: ../Images/arrow.png
    :align: center

.. image:: ../Images/main_diagram_step_disparity.png
    :align: center
    :target: step_by_step/disparity.html

.. image:: ../Images/arrow.png
    :align: center

.. image:: ../Images/main_diagram_step_refinement.png
    :align: left
    :target: step_by_step/refinement.html

.. image:: ../Images/main_diagram_step_filter.png
    :align: right
    :target: step_by_step/filtering.html

.. image:: ../Images/main_diagram_step_validation.png
    :align: center
    :target: step_by_step/validation.html

.. image:: ../Images/arrow.png
    :align: center

.. image:: ../Images/main_diagram_output.png
    :align: center
    :target: output.html

.. note::
    - Dark red blocks represent mandatory steps.
    - Pink blocks represent optional steps.

Configuration file
******************
The configuration file provides a list of parameters to Pandora so that the processing pipeline can
run according to the parameters choosen by the user.

Pandora works with JSON formatted data with the following nested structures.

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
        }
    }

All configuration parameters are described in :ref:`inputs` and :ref:`step_by_step` chapters.
As shown on the diagram, stereo steps must respect on order of priority, and can be called multiple times as explain on :ref:`sequencing` chapter.

.. note::
    The right disparity map can be computed if *right_disp_map* parameter is activated. See :ref:`outputs`.

Example
*******

1. Install

.. code-block:: bash

    pip install pandora[sgm]

2. Create a configuration file

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
            "right_disp_map":
            {
              "method": "accurate"
            },
            "matching_cost" :
            {
              "matching_cost": "census",
              "window_size": 5,
              "subpix": 1
            },
            "optimization" :
            {
              "optimization_method": "sgm",
              "P1": 8,
              "P2": 32,
            },

            "disparity":
            {
              "disparity_method": "wta",
              "invalid_disparity": "NaN"
            },
            "refinement":
            {
              "refinement_method": "vfit"
            },
            "filter" :
            {
              "filter_method": "median",
              "filter_size": 3
            },

            "validation" :
            {
              "validation_method": "cross_checking",
              "cross_checking_threshold": 1
            },
            "filter.after.validation" :
            {
              "filter_method": "median",
              "filter_size": 3
            }
        }
    }

3. Run Pandora

.. code-block:: bash

    pandora config.json output/
