.. _matching_cost:

Matching cost computation
=========================

Theoretical basics
------------------

The first step is to compute the cost volume containing the similarity coefficients.
Different measures of similarity are available in Pandora :

- SAD (Sum of Absolute Differences)
- SSD (Sum of Squared Differences)
- Census [Zabih1994]_
- ZNCC (Zero mean Normalized Cross Correlation)

It is possible to oversample the cost volume by a factor of 2 or 4 ( with the *subpix* parameter ) compared to
to the input images. It can be useful for :ref:`disparity_refinement`

.. note::  The cost volume disparity dimension is sampled at the input images rate by default.
           Thus disparities tested are integers. However, to prevent from aliasing effects when
           refining the disparity map, one can use the *subpix* parameter.
           This will add subpixel disparities into the cost volume by oversampling the disparity dimension by an even factor.


Pandora can take into account a mask and a nodata value for each image. The masks and nodata are used during
the matching cost computation  :

- Nodata pixel management: if the left window contains nodata, the center pixel of the window is invalidated.
  Therefore,the disparity range is invalidated : :math:`cost(x, y, \forall d) = nan`.
  If the right window contains nodata, the center pixel is invalidated. As a result, the pixels of the left image
  such as :math:`I_{L}(x, y) = I_{R}(x + d, y)`, are invalidated :math:`cost(x, y, d) = nan`


- Management of hidden pixels: if the center pixel of the left window is hidden, the disparity range is
  invalidated : :math:`cost(x, y, \forall d) = nan`.
  If the pixel in the center of the right window is hidden, the pixels of the left image such as
  :math:`I_{L}(x, y) = I_{R}(x + d, y)` are invalidated :math:`cost(x, y, d) = nan`

.. [Zabih1994] Zabih, R., & Woodfill, J. (1994, May). Non-parametric local transforms for computing visual correspondence.
       In European conference on computer vision (pp. 151-158). Springer, Berlin, Heidelberg.


Configuration and parameters
----------------------------

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Available value**,**Required**
    *matching_cost_method*,Similarity measure,string,,ssd sad census zncc  or mc_cnn if plugin_mccnn is installed,Yes
    *window_size*,Window size for similarity measure,int,5,Must be >0 and For "census" : {3 5 7 9 11 13},No
    *subpix*,Cost volume upsampling factor,int,1, {1 2 4},No
    *band*, Band value on which to calculate the correlation, str, None, band in left and right image's metadata,No
    *step*, Computing step for `pandora2d <https://pandora2d.readthedocs.io/en/latest/>`_, int, 1, 1, No

- For *mc_cnn* similarity measure see :ref:`plugin_mccnn_conf` of :ref:`plugin_mccnn` for sub-parameters and configuration example.

.. note::
    - NB: Parameter *band_im* must be present in the left and right image's metadata.
    - If the input images are multiband, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.

**Example for mono band images**

.. sourcecode:: json

    {
        "input" :
        {
            // ...
        },
        "pipeline" :
        {
            // ...
            "matching_cost":
            {
                "matching_cost_method": "ssd",
                "window_size": 7,
                "subpix": 4
            }
            // ...
        }
    }

**Example for multiband images**

.. sourcecode:: json

    {
        "input" :
        {
            // ...
        },
        "pipeline" :
        {
            // ...
            "matching_cost":
            {
                "matching_cost_method": "ssd",
                "band":"r",
                "window_size": 7,
                "subpix": 4
            }
            // ...
        }
    }
