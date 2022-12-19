.. _plugin_libsgm:

Plugin libSGM
=============

Theoretical basics
******************

`Pandora plugin <https://github.com/CNES/Pandora_plugin_libSGM>`_ to optimize the cost volume following SGM algorithm [Hirschmuller2008]_ with the `libSGM library <https://github.com/CNES/Pandora_libSGM>`_ .

As a reminder, SGM equation: :math:`E(D) = \sum_{p}{C(p,Dp)} + \sum_{q \in Np}{P_{1}T(|D_{p} - D_{q}|=1)} + \sum_{q \in Np}{P_{2}T(|D_{p} - D_{q}|>1)}`
with :math:`D` the disparity image and :math:`N_{p}` the neigborhood of :math:`p`.

One can implement their own penalty estimation methods, corresponding to :math:`P_{1}` and :math:`P_{2}` parameters of SGM equation.
Some are already avalaible,computed by the plugin_libsgm and divided in two categories:

1. Methods inspired by the one defined on [Hirschmuller2008]_ which are identified by *penalty_method=sgm_penalty*

    - Constant for :math:`P_{1}` and :math:`P_{2}`.
    - Methods depending on intensity gradient of the left image for :math:`P_{2}` estimation defined in [Banz2012]_.

        - The negative gradient: :math:`P_{2} = - \alpha \mid I(p)-I(p-r) \mid + \gamma \ ` with I for intensity on left image
        - The inverse gradient :math:`P_{2} = \frac{\alpha}{\mid I(p)-I(p-r) \mid + \beta} + \gamma \ ` with I for intensity on left image

2. Method defined by [Zbontar2016]_,depending on intensity gradient of left and right images which is identified by:

    - *penalty_method=mc_cnn_fast_penalty*, recommended when *mc_cnn_fast* matching cost method used.
    - *penalty_method=mc_cnn_accurate_penalty*,  recommended when *mc_cnn_fast* matching cost method used.

    For both of them, same equation but different default values for parameters as :math:`sgm_P1`, :math:`sgm_P2` ...

    .. math::
      D1 &= \mid I_{l}(p-d)-I_{l}(p-d-r) \mid \ , D2 = \mid I_{r}(p-d)-I_{r}(p-d-r) \mid \\
      P_1 &= sgm_P1 \ , P_2 = sgm_P2 \ if \ D1<sgm_D \ , D2<sgm_D \\
      P_1 &= \frac{sgm_P1}{sgm_Q2} \ , P_2 = \frac{sgm_P2}{sgm_Q2} \ if \ D1 \geq sgm_D \ , D2 \geq sgm_D \\
      P_1 &= \frac{sgm_P1}{sgm_Q1} \ , P_2 = \frac{sgm_P2}{sgm_Q1} \ otherwise

**Confidence**

The user can activate *use_confidence* if he wants to apply the confidence as follows:

    .. math::
      E(D) = \sum_{p}{C(p,Dp) * Confidence(p)} + \sum_{q \in Np}{P_{1}T(|D_{p} - D_{q}|=1)} + \sum_{q \in Np}{P_{2}T(|D_{p} - D_{q}|>1)}

with :math:`D` the disparity image and :math:`N_{p}` the neigborhood of :math:`p`.

The user must have computed ambiguity confidence previously in the pipeline. If not, default confidence values equal to 1 will be used, which is equivalent to not use confidence.

**Piecewise Optimization**

The user can activate the piecewise optimization by choosing the layer *piecewise_optimization_layer* to use as segments for piecewise optimization.
For each segment, optimization will only be applied inside this segment.

The user can use the `classif` or `segm` layer respectively corresponding to the `left_classif` and `left_segm` (`right_classif` and `right_segm` for right image) specified in the input configuration.

The input segmentation or classification .tif file must be the same format as the input image, with the same dimensions. Moreover, this option requires :
    - All pixels inside a segment must have the same value (int or float).
    - The value of a class or a segment must be different to the values of surrounding classes or segments.
    - The data must be dense : for example if the user wants to perform a piecewise optimization with only one small segment, the data must be composed of two different values.


The following diagram explains the concept:

    .. image:: ../../Images/piecewise_optimization_segments.png

.. [Hirschmuller2008] H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166
.. [Banz2012] Banz, C. & Pirsch, P. & Blume, Holger. (2012). EVALUATION OF PENALTY FUNCTIONS FOR SEMI-GLOBAL MATCHING COST AGGREGATION. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. XXXIX-B3. 1-6. 10.5194/isprsarchives-XXXIX-B3-1-2012.
.. [Zbontar2016] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.

.. _plugin_libsgm_conf:

Configuration and parameters
****************************

+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| Name                         | Description                                             | Type   | Default value | Available value                                                | Required                                             |
+==============================+=========================================================+========+===============+================================================================+======================================================+
| penalty_method               | Method for penalty estimation                           | string | "sgm_penalty" | "sgm_penalty","mc_cnn_fast_penalty", "mc_cnn_accurate_penalty" | No                                                   |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| p2_method                    | sub-method of *sgm_penalty* for P2 penalty estimation   | String | "constant"    | "constant" , "negativeGradient", "inverseGradient"             | No. Only available if *penalty_method = sgm_penalty* |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| overcounting                 | overcounting correction                                 | Boolean| False         | True, False                                                    | No                                                   |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| min_cost_paths               | Number of sgm paths that give the same final disparity  | Boolean| False         | True, False                                                    | No                                                   |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| use_confidence               | Apply confidence to cost volume                         | Boolean| False         | True, False                                                    | No                                                   |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+
| piecewise_optimization_layer | Layer to use during piecewise optimization              | string | "None"        | "None", "classif", "segm"                                      | No                                                   |
+------------------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------------------+------------------------------------------------------+

There are some parameters depending on penalty_method choice and p2_method choice.

- *penalty_method = sgm_penalty* and  *p2_method = constant*

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

.. note::  The default values are intended for use with Census matching cost method. We cannot say that they are suitable with other matching cost method.

- *penalty_method = sgm_penalty* and *p2_method = negativeGradient*

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| alpha | Penalty parameter | float        | 1.0           |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| gamma | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- *penalty_method = sgm_penalty* and *p2_method = inverseGradient*

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| alpha | Penalty parameter | float        | 1.0           |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| beta  | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| gamma | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- *penalty_method = mc_cnn_fast_penalty*

+------+-------------------+--------------+---------------+-----------------+----------+
| Name | Description       | Type         | Default value | Available value | Required |
+======+===================+==============+===============+=================+==========+
| P1   | Penalty parameter | int or float | 2.3           | >0              | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| P2   | Penalty parameter | int or float | 55.9          | P2 > P1         | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q1   | Penalty parameter | int or float | 4             |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q2   | Penalty parameter | int or float | 2             |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| D    | Penalty parameter | int or float | 0.08          |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| V    | Penalty parameter | int or float | 1.5           |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+

.. note:: P1, P2, Q1, Q2, D, V represent sgm_P1, sgm_P2, sgm_Q1, smg_Q2, sgm_D, sgm_V respectively

- *penalty_method = mc_cnn_accurate_penalty*

+------+-------------------+--------------+---------------+-----------------+----------+
| Name | Description       | Type         | Default value | Available value | Required |
+======+===================+==============+===============+=================+==========+
| P1   | Penalty parameter | int or float | 1.3           | >0              | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| P2   | Penalty parameter | int or float | 18.1          | P2 > P1         | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q1   | Penalty parameter | int or float | 4.5           |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q2   | Penalty parameter | int or float | 9             |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| D    | Penalty parameter | int or float | 0.13          |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| V    | Penalty parameter | int or float | 2.75          |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+


**Example**

.. sourcecode:: text

    {
      "input" : {
            ...
      },
      "pipeline" :
       {
            ...
            "optimization": {
                "optimization_method": "sgm",
                "penalty_method": "sgm_penalty",
                "P1": 4,
                "P2": 20
            }
            ...
        }
    }


Pandora's data
**************

As a reminder, Pandora generates a cost volume, during the matching cost computation step. This cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.

The plugin receives this cost volume and uses the libsgm to optimize it. Then, this optimized cost volume is returned
to Pandora.

Moreover, if *cost_min_path* option is activated, the cost volume is enriched with a new confidence_measure called
*optimization_pluginlibSGM_nbOfDisp*. This 2-dimension map represents the number of sgm paths that give the same
position for minimal optimized cost at each point.
