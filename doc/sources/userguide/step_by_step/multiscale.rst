.. _multiscale:

Multiscale
=====================

Theoretical basics
------------------


If multiscale process is selected, the whole pipeline is repeated for each scale.

An image pyramid is computed according to the given scale factor and number of scales, and pyramid images are processed from coarse to fine.

The disparity map of the previous pipeline is used to construct two disparity grids, one for minimum disparity and one for maximum disparity, representing the disparity range per pixel. And then these two grids are zoomed and multiplied for the next scale.

- If the pixel was invalid in the previous scale, consider the whole disparity range.

- Add a disparity marge on the maximum and minimum resulting values to ensure the disparity range width.

- If the pixel was valid in the previous scale, consider the maximum and minimum disparity values of the valid pixels in the pixel window.

.. note::
  The invalid pixels of the full resolution image are interpolated using the method proposed in [Hirschmuller2007]_ before the Gaussian filtering and decimation to avoid its spreading.
  However, the full resolution image of the pyramid will be the original image with the original invalid pixels.

.. [Hirschmuller2007] HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.


Configuration and parameters
----------------------------

+---------------------+-------------------------------------------------------+------------+---------------+------------------------+----------+
| Name                | Description                                           | Type       | Default value | Available value        | Required |
+=====================+=======================================================+============+===============+========================+==========+
| *multiscale_method* | Multiscale method name                                | string     |               | "fixed_zoom_pyramid"   | Yes      |
+---------------------+-------------------------------------------------------+------------+---------------+------------------------+----------+
| *num_scales*        | Number of scales to process                           | int        |  2            | >= 2                   | No       |
+---------------------+-------------------------------------------------------+------------+---------------+------------------------+----------+
| *scale_factor*      | Scale factor by which reduce the image between scales | int        |  2            | >= 2                   | No       |
+---------------------+-------------------------------------------------------+------------+---------------+------------------------+----------+
| *marge*             | Marge to avoid zero disparity range                   | int        |  1            | >= 0                   | No       |
+---------------------+-------------------------------------------------------+------------+---------------+------------------------+----------+

.. note::
  Multiscale with a num_scales = 1 cannot be chosen. For implementation without multiscale processing, do not add this entry in the pipeline configuration.

.. note::
  Multiscale method cannot be chosen if disparity maps are grids.

**Example**

.. sourcecode:: text

    {
      "input" : {
            ...
      },
      "pipeline" :
       {
            ...
            "multiscale": {
                "multiscale _method": "fixed_zoom_pyramid",
                "num_scales": 3,
                "marge": 2
            }
        }
    }
