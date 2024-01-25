.. _plugin_mccnn:

Plugin mccnn
=============

Theoretical basics
******************

`Pandora plugin <https://github.com/CNES/Pandora_plugin_mccnn>`_ to compute the cost volume using the similarity measure produced by mc-cnn neural network [MCCNN]_, with the  `MCCNN library <https://github.com/CNES/Pandora_MCCNN>`_ .


As a reminder, mc-cnn is a neural network which computes a similarity measure on pair of small image patches. This similarity measure is computed between each possible patch to initialize the cost volume.
Two different architectures of mc-cnn exist : mc-cnn fast and mc-cnn accurate. 

Only the mc-cnn fast architecture is available in this plugin because processing time of mc-cnn accurate is too long for 
practical use (about 80 times longer than mc-cnn fast). Moreover, the improvement on the results of mc-cnn accurate is usually small. 

Figure below details the architecture of mc-cnn fast.


.. figure:: ../../Images/mc_cnn_fast_architecture.png

|

Pretrained weights for mc-cnn fast network are available in the MC-CNN repository https://gitlab.cnes.fr/3d/PandoraBox/mc-cnn :

-  mc_cnn_fast_mb_weights.pt are the weights of the pretrained networks on the Middlebury dataset [Middlebury]_
-  mc_cnn_fast_data_fusion_contest.pt are the weights of the pretrained networks on the Data Fusion Contest dataset [DFC]_


.. [MCCNN] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.
.. [Middlebury] Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014, September). High-resolution stereo datasets with subpixel-accurate ground truth. In German conference on pattern recognition (pp. 31-42). Springer, Cham.
.. [DFC] Bosch, M., Foster, K., Christie, G., Wang, S., Hager, G. D., & Brown, M. (2019, January). Semantic stereo for incidental satellite images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1524-1532). IEEE.

.. _plugin_mccnn_conf:

Configuration and parameters
****************************

+------------------------+------------------------------------+--------+---------------------------+---------------------+----------+
| Name                   | Description                        | Type   | Default value             | Available value     | Required |
+========================+====================================+========+===========================+=====================+==========+
| *matching_cost_method* | Similarity measure                 | string |                           | "mc_cnn"            | Yes      |
+------------------------+------------------------------------+--------+---------------------------+---------------------+----------+
| *model_path*           | Path to the pretrained network     | string | mc_cnn_fast_mb_weights.pt |                     | No       |
+------------------------+------------------------------------+--------+---------------------------+---------------------+----------+
| *window_size*          | Window size for similarity measure | int    | 11                        | {11}                | No       |
+------------------------+------------------------------------+--------+---------------------------+---------------------+----------+
| *subpix*               | Cost volume upsampling factor      | int    | 1                         | {1}                 | No       |
+------------------------+------------------------------------+--------+---------------------------+---------------------+----------+

.. note::  Window size for mc-cnn similarity measure is fixed to 11 and cannot be changed.

.. note::  It is not possible to upsampled the cost volume with the mc-cnn similarity measure, therefore the subpixel parameter must be 1.


**Example**

.. sourcecode:: json

    {
        "input" :
        {
            //...
        },
        "pipeline" :
        {
            // ...
            "matching_cost":
            {
                "matching_cost_method": "mc_cnn",
                "model_path": "plugin_mc-cnn/weights/mc_cnn_fast_mb_weights.pt",
                "window_size": 11,
                "subpix": 1
            }
            // ...
        }
    }


Pandora's data
**************

The plugin generates a cost volume, during the matching cost computation step. This cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.
