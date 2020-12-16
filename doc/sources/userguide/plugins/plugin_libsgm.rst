.. _plugin_libsgm:

Plugin libSGM
=============

Theoretical basics
******************

`Pandora plugin <https://github.com/CNES/Pandora_plugin_libSGM>`_ to optimize the cost volume following SGM algorithm [Hirschmuller2008]_ with the `libSGM library <https://github.com/CNES/Pandora_libSGM>`_ .

One can implement their own penalty estimation methods, corresponding to P1 and P2 parameters of SGM equation.
Some are already avalaible and computed by the plugin_libsgm:

* Methods depending on intensity gradient of the left image [Banz2012]_.
* Method depending on intensity gradient of left and right image [Zbontar2016]_.

.. [Hirschmuller2008] H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166
.. [Banz2012] Banz, C. & Pirsch, P. & Blume, Holger. (2012). EVALUATION OF PENALTY FUNCTIONS FOR SEMI-GLOBAL MATCHING COST AGGREGATION. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. XXXIX-B3. 1-6. 10.5194/isprsarchives-XXXIX-B3-1-2012.
.. [Zbontar2016] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.

.. _plugin_libsgm_conf:

Configuration and parameters
****************************

+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| Name               | Description                                             | Type   | Default value | Available value                                    | Required                                                 |
+====================+=========================================================+========+===============+====================================================+==========================================================+
| penalty_estimation | Method for penalty estimation                           | string | "sgm_penalty" | "sgm_penalty","mc_cnn_penalty"                     | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| p2_method          | Method for p2 penalty estimation                        | String | "constant"    | "constant" , "negativeGradient", "inverseGradient" | No. Only available if penalty_estimation = "sgm_penalty" |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| overcounting       | overcounting correction                                 | Boolean| False         | True, False                                        | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| min_cost_paths     | Number of sgm paths that give the same final disparity  | Boolean| False         | True, False                                        | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+

There are some parameters depending on penalty_estimation choice and p2_method choice.

- penalty_estimation = "sgm_penalty" and  p2_method = "constant"

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- penalty_estimation = "sgm_penalty" and p2_method = "negativeGradient"

:math:`P2 = - \alpha \mid I(p)-I(p-r) \mid + \gamma \ ` with I for intensity on left image

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

- penalty_estimation = "sgm_penalty" and p2_method = "inverseGradient"

:math:`P2 = \frac{\alpha}{\mid I(p)-I(p-r) \mid + \beta} + \gamma \ ` with I for intensity on left image

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

- penalty_estimation = "mc_cnn_penalty"

.. math::
  D1 &= \mid I_{l}(p-d)-I_{l}(p-d-r) \mid \ , D2 = \mid I_{r}(p-d)-I_{r}(p-d-r) \mid \\
  P1 &= sgm_P1 \ , P2 = sgm_P2 \ if \ D1<sgm_D \ , D2<sgm_D \\
  P1 &= \frac{sgm_P1}{sgm_Q2} \ , P2 = \frac{sgm_P2}{sgm_Q2} \ if \ D1 \geq sgm_D \ , D2 \geq sgm_D \\
  P1 &= \frac{sgm_P1}{sgm_Q1} \ , P2 = \frac{sgm_P2}{sgm_Q1} \ otherwise

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

.. note:: P1, P2, Q1, Q2, D, V represent sgm_P1, sgm_P2, sgm_Q1, smg_Q2, sgm_D, sgm_V respectively

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
                "penalty_estimation": "sgm_penalty",
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
