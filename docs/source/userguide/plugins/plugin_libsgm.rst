.. _plugin_libsgm:

Plugin libSGM
=============

Theoretical basics
******************

`Pandora plugin <https://github.com/CNES/Pandora_plugin_libSGM>`_ to optimize the cost volume following SGM algorithm [Hirschmuller2008]_ with the `libSGM library <https://github.com/CNES/Pandora_libSGM>`_ .

As a reminder, SGM equation: :math:`E(D) = \sum_{p}{C(p,Dp)} + \sum_{q \in Np}{P_{1}T(|D_{p} - D_{q}|=1)} + \sum_{q \in Np}{P_{2}T(|D_{p} - D_{q}|>1)}`
with


:math:`D` the disparity image and :math:`N_{p}` the neighborhood of :math:`p`.

One can implement their own penalty estimation methods, corresponding to :math:`P_{1}` and :math:`P_{2}` parameters of SGM equation.
Some are already available,computed by the plugin_libsgm and divided in two categories:

1. Methods inspired by the one defined on [Hirschmuller2008]_ which are identified by *penalty_method=sgm_penalty*

    - Constant for :math:`P_{1}` and :math:`P_{2}`.
    - Methods depending on intensity gradient of the left image for :math:`P_{2}` estimation defined in [Banz2012]_.

        - The negative gradient: :math:`P_{2} = - \alpha \mid I(p)-I(p-r) \mid + \gamma \ ` with I for intensity on left image
        - The inverse gradient :math:`P_{2} = \frac{\alpha}{\mid I(p)-I(p-r) \mid + \beta} + \gamma \ ` with I for intensity on left image

2. Method defined by [Zbontar2016]_, depending on intensity gradient of left and right images which is identified by *penalty_method=mc_cnn_fast_penalty*

    Same equation but different default values for parameters as :math:`sgm_P1`, :math:`sgm_P2` ...

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

Method defined by [Dumas2022]_. The user can activate the piecewise optimization by choosing the layer *geometric_prior* to use as segments for piecewise optimization.
For each segment, optimization will only be applied inside this segment.

If no *geometric_prior* is specified, the default `internal` mode is used. For now, 3SGM doesn't compute piecewise layer from internal mode.
Hence, no piecewise optimization will be done (equivalent to performing SGM optimization).

The user can use the `classif` or `segm` layer respectively corresponding to the `left_classif` and `left_segm` (`right_classif` and `right_segm` for right image) specified in the input configuration.

The input segmentation or classification must be provided as raster image. This .tif file must have the same format as the input image, with the same dimensions. Moreover, this input must meet the following conditions :
  For input segmentation:
    - All pixels inside a segment must have the same value (int or float).
    - The value of a segment must be different to the values of surrounding segments.
    - The data must be dense, which means that all pixels must be filed in: for example if the user wants to perform a piecewise optimization with only one small segment, the data must be composed of two different values for all the image.
  For input classification:
    - The input classification image must have one band per class (with value 1 on the pixels belonging to the class, and 0 for the rest), and the band's names must be present on the image metadata. To see how to add band's names on the classification image's metadata, please
      see :ref:`faq`.


The following diagram explains the concept:

    .. image:: ../../Images/piecewise_optimization_segments.png

.. [Hirschmuller2008] Hirschmuller, H. "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166
.. [Banz2012] Banz, C. & Pirsch, P. & Blume, Holger. (2012). EVALUATION OF PENALTY FUNCTIONS FOR SEMI-GLOBAL MATCHING COST AGGREGATION. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. XXXIX-B3. 1-6. 10.5194/isprsarchives-XXXIX-B3-1-2012.
.. [Zbontar2016] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.
.. [Dumas2022] Dumas, L., Defonte, V., Steux, Y., and Sarrazin, E.: IMPROVING PAIRWISE DSM WITH 3SGM: A SEMANTIC SEGMENTATION FOR SGM USING AN AUTOMATICALLY REFINED NEURAL NETWORK, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-2-2022, 167–175, https://doi.org/10.5194/isprs-annals-V-2-2022-167-2022, 2022.

.. _plugin_libsgm_conf:

Configuration and parameters
****************************

There are some parameters depending on sgm but not penalties


.. list-table:: 
   :widths: 19 19 19 19 19 19
   :header-rows: 1

   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - *overcounting*
     - Overcounting correction
     - Boolean
     - False
     - | True or 
       | False
     - No
   * - *min_cost_paths*
     - | Number of sgm paths that give the same
       | final disparity
     - Boolean
     - False
     - | True or 
       | False
     - No
   * - *use_confidence*
     - Apply ambiguity confidence to cost volume
     - string
     - None
     - 
     - No
   * - *geometric_prior source*
     - Layer to use during piecewise optimization
     - dict
     - "internal"
     - | "internal" or 
       | "classif" or
       | "segm"
     - No
   * - *geometric_prior classes*
     - Classes to use if source is classif
     - list
     - 
     - 
     - Only if source is "classif"
   * - *penalty*
     - | Dictionary containing all parameters 
       | related to penalties
     - dict
     - | {"penalty_method": "sgm_penalty"
       | "P1": 4 
       | "P2": 20}
     - *cf. following tables*
     - Only if source is "classif"


Penalty configuration
#####################

.. list-table:: Configuration for penalty estimation
   :widths: 19 19 19 19 19 19
   :header-rows: 1

   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - penalty_method
     - Method for penalty estimation
     - string
     - "sgm_penalty"
     - | "sgm_penalty" or 
       | "mc_cnn_fast_penalty"
     - No
   * - p2_method
     - Sub-method of *sgm_penalty* for P2 penalty estimation
     - string
     - "constant"
     - | "constant" or
       | "negativeGradient" or
       | "inverseGradient"
     - | No. 
       | Only available if *penalty_method = sgm_penalty*

There are some parameters depending on penalty_method choice and p2_method choice.

.. tabs::

    .. tab:: sgm_penalty

        .. tabs:: 
            
            .. tab:: constant

                .. list-table:: 
                    :widths: 19 19 19 19 19 19
                    :header-rows: 1

                    * - Name
                      - Description
                      - Type
                      - Default value
                      - Available value
                      - Required
                    * - P1
                      - Penalty parameter
                      - int or float
                      - 8
                      - >0
                      - No
                    * - P2
                      - Penalty parameter
                      - int or float
                      - 32
                      - P2 > P1
                      - No
                
                .. note::  The default values are intended for use with Census matching cost method. We cannot say that they are suitable with other matching cost method.

            .. tab:: negativeGradient

                .. list-table:: 
                    :widths: 19 19 19 19 19 19
                    :header-rows: 1

                    * - Name
                      - Description
                      - Type
                      - Default value
                      - Available value
                      - Required
                    * - P1
                      - Penalty parameter
                      - int or float
                      - 8
                      - >0
                      - No
                    * - P2
                      - Penalty parameter
                      - int or float
                      - 32
                      - P2 > P1
                      - No
                    * - alpha
                      - Penalty parameter
                      - float
                      - 1.0
                      - 
                      - No
                    * - gamma
                      - Penalty parameter
                      - int or float
                      - 1
                      - 
                      - No

            .. tab:: inverseGradient

                .. list-table:: 
                    :widths: 19 19 19 19 19 19
                    :header-rows: 1

                    * - Name
                      - Description
                      - Type
                      - Default value
                      - Available value
                      - Required
                    * - P1
                      - Penalty parameter
                      - int or float
                      - 8
                      - >0
                      - No
                    * - P2
                      - Penalty parameter
                      - int or float
                      - 32
                      - P2 > P1
                      - No
                    * - alpha
                      - Penalty parameter
                      - float
                      - 1.0
                      - 
                      - No
                    * - beta
                      - Penalty parameter
                      - int or float
                      - 1
                      - 
                      - No
                    * - gamma
                      - Penalty parameter
                      - int or float
                      - 1
                      - 
                      - No

    
    .. tab:: mc_cnn_fast_penalty

        .. list-table:: 
            :widths: 19 19 19 19 19 19
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Default value
              - Available value
              - Required
            * - P1
              - Penalty parameter
              - int or float
              - 2.3
              - >0
              - No
            * - P2
              - Penalty parameter
              - int or float
              - 55.9
              - P2 > P1
              - No
            * - Q1
              - Penalty parameter
              - int or float
              - 4
              - 
              - No
            * - Q2
              - Penalty parameter
              - int or float
              - 2
              - 
              - No
            * - D
              - Penalty parameter
              - int or float
              - 0.08
              - 
              - No
            * - V
              - Penalty parameter
              - int or float
              - 1.5
              - 
              - No


**Example using sgm optimization**

.. sourcecode:: json

    {
      "input" : {
            // ...
      },
      "pipeline" :
       {
            // ...
            "optimization": {
                "optimization_method": "sgm",
                "penalty": {
                    "penalty_method": "sgm_penalty",
                    "P1": 4,
                    "P2": 20
                }
            }
            // ...
        }
    }


**Example using 3sgm optimization and geometric_prior classif**

.. sourcecode:: json

    {
      "input" : {
            "left": {
                "img": "PATH",
                "classif": "PATH"
            },
            "right": {
                "img": "PATH"
            }
      },
      "pipeline" :
       {
            // ...
            "optimization": {
                "optimization_method": "3sgm",
                "penalty": {
                    "penalty_method": "sgm_penalty",
                    "P1": 4,
                    "P2": 20
                },
                "geometric_prior": {"source": "classif",
                                    "classes": ["roads", "buildings"]
                               },
            }
            // ...
        }
    }

.. warning:: If no semantic segmentation step was computed before 3SGM optimization in the pipeline, internal segmentation will be the default value.

Pandora's data
**************

As a reminder, Pandora generates a cost volume, during the matching cost computation step. This cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.

The plugin receives this cost volume and uses the libsgm to optimize it. Then, this optimized cost volume is returned
to Pandora.

Moreover, if *cost_min_path* option is activated, the cost volume is enriched with a new confidence_measure called
*optimization_plugin_libsgm_nb_of_directions*. This 2-dimension map represents the number of sgm paths that give the same
position for minimal optimized cost at each point.
