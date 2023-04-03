.. _semantic_segmentation:

Semantic segmentation
=====================

Theoretical basics
------------------

The main idea of our proposal is to create a building semantic segmentation from the left epipolar image and use it to optimize the Disparity Space Image (DSI).
We then propose a slight modification of SGM optimization method to incorporate the semantic segmentation. This methodology is resumed in the following equation.
We simply stop the optimization of a given path every time this path crosses a building edge

:math:`E(D) = \sum_{p}{C(p,Dp)} + \beta(\sum_{q \in Np}{P_{1}T(|D_{p} - D_{q}|=1)} + \sum_{q \in Np}{P_{2}T(|D_{p} - D_{q}|>1)})`
with :math:`D` the disparity image, :math:`N_{p}` the neighborhood of :math:`p` and :math:`\beta` stops optimization of a given path every time this path crosses a building edge.

OSM labels or Neural network generated labels can be used to define edges.

The method available in Pandora is

- ARNN, made available by :ref:`plugin_arnn`.


Configuration and parameters
----------------------------

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Available value**,**Required**
    *segmentation_method*,Semantic segmentation method,str,ARNN,ARNN,Yes

Optimization method:

- *ARNN*: :ref:`plugin_arnn_conf` of :ref:`plugin_arnn` for sub-parameters and configuration example.
