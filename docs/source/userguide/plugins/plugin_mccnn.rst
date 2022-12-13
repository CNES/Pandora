.. _plugin_mccnn:

Plugin mccnn
=============

Theoretical basics
******************

`Pandora plugin <https://github.com/CNES/Pandora_plugin_mccnn>`_ to compute the cost volume using the similarity measure produced by mc-cnn neural network [MCCNN]_, with the  `MCCNN library <https://github.com/CNES/Pandora_MCCNN>`_ .


As a reminder, mc-cnn is a neural network which computes a similarity measure on pair of small image patches. This similarity measure is computed between each possible patch to initialize the cost volume.
There are two architectures: mc-cnn fast and mc-cnn accurate, figures  below detail the networks. Both networks  have the same input and output, the mc-cnn fast network is faster than the mc-cnn accurate network.
The fast architecture uses a fixed similarity measure (dot product) while the accurate architecture attempts to learn a similarity measure.


   .. figure:: ../../Images/mc_cnn_architectures.svg

      Left : mc-cnn fast architecture. Right : mc-cnn fast architecture


Pretrained weights for mc-cnn fast and mc-cnn accurate neural networks are available in the `Plugin_mccnn repository <https://github.com/CNES/Pandora_plugin_mccnn>`_ :

-  mc_cnn_fast_mb_weights.pt and mc_cnn_accurate_mb_weights.pt are the weights of the pretrained networks on the Middlebury dataset [Middlebury]_
-  mc_cnn_fast_data_fusion_contest.pt and mc_cnn_accurate_data_fusion_contest.pt are the weights of the pretrained networks on the Data Fusion Contest dataset [DFC]_


.. [MCCNN] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.
.. [Middlebury] Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014, September). High-resolution stereo datasets with subpixel-accurate ground truth. In German conference on pattern recognition (pp. 31-42). Springer, Cham.
.. [DFC] Bosch, M., Foster, K., Christie, G., Wang, S., Hager, G. D., & Brown, M. (2019, January). Semantic stereo for incidental satellite images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1524-1532). IEEE.

.. _plugin_mccnn_conf:

Configuration and parameters
****************************

+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+
| Name                   | Description                        | Type   | Default value | Available value                | Required |
+========================+====================================+========+===============+================================+==========+
| *matching_cost_method* | Similarity measure                 | string |               | "mc_cnn"                       | Yes      |
+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *mc_cnn_arch*          | mc-cnn architecture                | string |               | "fast", "accurate"             | Yes      |
+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *model_path*           | Path to the pretrained network     | string |               |                                | Yes      |
+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *window_size*          | Window size for similarity measure | int    | 11            | {11}                           | No       |
+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+
| *subpix*               | Cost volume upsampling factor      | int    | 1             | {1}                            | No       |
+------------------------+------------------------------------+--------+---------------+--------------------------------+----------+

.. note::  Window size for mc-cnn similarity measure is fixed to 11 and cannot be changed.

.. note::  It is not possible to upsampled the cost volume with the mc-cnn similarity measure, therefore the subpixel parameter must be 1.


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
            "matching_cost":
            {
                "matching_cost_method": "mc_cnn",
                "mc_cnn_arch": "fast"
                "model_path": "plugin_mc-cnn/weights/mc_cnn_fast_mb_weights.pt"
                "window_size": 11,
                "subpix": 1
            }
            ...
        }
    }


Pandora's data
**************

The plugin generates a cost volume, during the matching cost computation step. This cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.
