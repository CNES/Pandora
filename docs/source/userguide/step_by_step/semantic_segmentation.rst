.. _semantic_segmentation:

Semantic segmentation
=====================

Theoretical basics
------------------

The main idea of our proposal is to create a semantic segmentation map to improve the quality of the disparity map by using this information in other steps of the pipeline.
The semantic segmentation produced during this step can be used during the SGM optimization step to improve results. The methodology is resumed in the following equation:


:math:`E(D) = \sum_{p}{C(p,Dp)} + \beta(\sum_{q \in Np}{P_{1}T(|D_{p} - D_{q}|=1)} + \sum_{q \in Np}{P_{2}T(|D_{p} - D_{q}|>1)})`
with :math:`D` the disparity image, :math:`N_{p}` the neighborhood of :math:`p` and :math:`\beta` stops optimization of a given path every time this path crosses a building edge.

The method available in Pandora is

- ARNN for building semantic segmentation, made available by :ref:`plugin_arnn`.


Configuration and parameters
----------------------------

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Available value**,**Required**
    *segmentation_method*,Semantic segmentation method,str,ARNN,ARNN,Yes

Optimization method:

- *ARNN*: :ref:`plugin_arnn_conf` of :ref:`plugin_arnn` for sub-parameters and configuration example.
