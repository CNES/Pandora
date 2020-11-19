Pandora's data
==============

Images
-----------

Pandora reads input images before stereo computation and creates two dataset for left and right
images containing data's image, data's mask and additionnal information.

Example of an image dataset

::

    Dimensions:  (col: 450, row: 375)
    Coordinates:
      * col      (col) int64 0 1 2 3 4 5 6 7 8 ... 442 443 444 445 446 447 448 449
      * row      (row) int64 0 1 2 3 4 5 6 7 8 ... 367 368 369 370 371 372 373 374
    Data variables:
        im       (row, col) float32 88.0 85.0 84.0 83.0 ... 176.0 180.0 165.0 172.0
        msk      (row, col) int16 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0
    Attributes:
        no_data_img:   0
        valid_pixels:  0
        no_data_mask:  1

.. note::
    This example comes from a dataset created by Pandora's reading function. Dataset attributes
    'valid_pixels' and 'no_data_mask' cannot be modified with this function. Its indicate the msk
    data convention.
    For API user who wants to create own dataset, without using Pandora's reading function, it is
    possible to declare its own mask convention with these attributes.


Cost volume
-----------

Pandora generates a cost volume during the first step: *Matching cost computation*. The cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.
When matching is impossible, the matching cost is np.nan.

This Dataset also has a :

- xarray.DataArray 3D confidence_measure, which contains the standard deviation of pixel intensity indicator.
  More specifically, this indicator represents standard deviation of pixel intensity of the left image
  inside a window (the same as the window used for matching cost)
  It is generated during the first step: *Matching cost computation*.
- xarray.DataArray disp_indices, which contains the minimum cost indices calculated in step *Disparity computation*..


Example of a cost volume


::

    <xarray.Dataset>
    Dimensions:       (col: 996, disp: 64, indicator: 1, row: 996)
    Coordinates:
      * row           (row) int64 2 3 4 5 6 7 8 9 ... 991 992 993 994 995 996 997
      * col           (col) int64 2 3 4 5 6 7 8 9 ... 991 992 993 994 995 996 997
      * disp          (disp) int64 -30 -29 -28 -27 -26 -25 -24 ... 28 29 30 31 32 33
      * indicator     (indicator) object 'stereo_pandora_intensityStd'
    Data variables:
        cost_volume   (row, col, disp) float32 nan nan nan nan ... nan nan nan nan
        confidence_measure   (row, col, indicator) float32 nan nan nan nan ... nan nan nan nan
        disp_indices  (row, col) float32 10.0 10.0 10.0 10.0 ... -10.0 -9.0 -10.0
    Attributes:
        measure:         census
        subpixel:        1
        offset_row_col:  2
        window_size:     5
        type_measure:    min
        cmax:            24
        optimization:    sgm

The cost volume corresponds to the variable cv ( and cv_right for the right / left cost volume ) in the file pandora/__init__.py :

.. sourcecode:: python

    def run(img_left, img_right, disp_min, disp_max, cfg, path_left=None, path_right=None):
        ...
        # Matching cost computation
        print('Matching cost computation...')
        cv = stereo_.compute_cost_volume(pandora_left, pandora_right, disp_min, disp_max)
        ...
        print('Computing the right disparity map with the accurate method...')
        cv_right = stereo_.compute_cost_volume(pandora_right, pandora_left, -disp_max, -disp_min)


.. note::

    The cost volume contains only the similarity factors calculated with the steps *Calculation of mapping costs*,
    *Aggregation of costs*, *Optimization*. It does not contain the interpolated factors ( calculated in step
    *disparity refinement*), these are available in the *interpolated_coeff* variable in the Disparity Dataset.


Disparity map
-------------

The *Disparity computation* step generates a disparity map in cost volume geometry. This disparity map is
a float32 type 2D xarray.DataArray, stored in a xarray.Dataset.
This Dataset also has a :

- xarray.DataArray 3D confidence_measure, which contains quality indicators. It can be enriched by indicators calculated in the different plugins.

  - standard deviation of the intensities within the a window: "stereo_pandora_intensityStd"
  - distance between left-right (or right-left) disparities: "validation_pandora_distanceOfDisp", if "cross_checking" validation is enabled
- xarray.DataArray validity_mask which represents the :ref:`validity_mask_data`.
- xarray.DataArray interpolated_coeff, which contains the similarity coefficients interpolated by the Disparity Refinement Method.


.. sourcecode:: text

    <xarray.Dataset>
    Dimensions:             (col: 1000, indicator: 2, row: 1000)
    Coordinates:
      * row                 (row) int64 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
      * col                 (col) int64 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
      * indicator           (indicator) object 'stereo_pandora_intensityStd' 'validation_pandora_distanceOfDisp'
    Data variables:
        disparity_map       (row, col) float32 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        validity_mask       (row, col) uint16 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1
        interpolated_coeff  (row, col) float64 nan nan nan nan ... nan nan nan nan
        confidence_measure  (row, col, indicator) float32 nan nan nan ... nan nan nan
    Attributes:
        measure:                census
        subpixel:               1
        offset_row_col:         0
        window_size:            5
        type_measure:           min
        cmax:                   24
        optimization:           sgm
        disp_min:               -30
        disp_max:               33
        refinement:             vfit
        filter:                 median
        validation:             cross_checking
        interpolated_disparity: none


The disparity maps correspond to the variables left, right in the pandora file __init__.py:

.. sourcecode:: python

    def run(img_left, img_right, disp_min, disp_max, cfg, path_left=None, path_right=None):
        ...
        # Disparity computation and validity mask
        print('Disparity computation...')
        left = disparity.to_disp(cv)
        ...
        return left, right

.. _validity_mask_data:

Validity mask
-------------

Validity masks are 2D xarray.DataArray and are 16-bit encoded: each bit represents a
rejection criterion (= 1 if rejection, = 0 otherwise):

 +---------+--------------------------------------------------------------------------------------------------------+
 | **Bit** | **Description**                                                                                        |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    0    |   * border of left image                                                                               |
 |         |   * nodata of left image                                                                               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |         | The point is invalid, there are two possible cases:                                                    |
 |         |                                                                                                        |
 |    1    |   - Disparity range does not permit to find any point on the right image                               |
 |         |   - nodata of right image                                                                              |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    2    | Information : disparity range cannot be used completely , reaching border of right image               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    3    | Information: calculations stopped at the pixel stage, sub-pixel interpolation was not successful       |
 |         | (for vfit: pixels d-1 and/or d+1 could not be calculated)                                              |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    4    | Information : closed occlusion                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    5    | Information : closed mismatch                                                                          |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    6    | The point is invalid: invalidated by the validity mask associated to the left image                    |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    7    | The point is invalid: right positions to be scanned invalidated by the mask of the right image         |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    8    | The Point is invalid: point located in an occlusion zone                                               |
 +---------+--------------------------------------------------------------------------------------------------------+
 |    9    | The point is invalid: mismatch                                                                         |
 +---------+--------------------------------------------------------------------------------------------------------+

The validity masks are stored in the xarray.Dataset left and right in the pandora/__init__.py file.

.. _border_management:

Border management
-----------------

Left image
^^^^^^^^^^^^^^^

Pixels of the left image for which the measurement thumbnail protrudes from the left image are truncated
in the cost volume, disparity maps and masks. Therefore, the memory occupancy of the cost volume is
diminished.
For a similarity measurement with a 5x5 window, these incalculable pixels in the left image correspond
to a 2-pixel crown at the top, bottom, right and left, and are represented by the offset_row_col attribute in
the xarray.Dataset. For an image of 100x100 with a window of 5x5, the products will be of dimension :

.. sourcecode:: text

   <xarray.Dataset>
   Dimensions:      (col: 96, row: 96)
   Coordinates:
     * row          (row) int64 2 3 4 5 6 7 8 9 10 ... 89 90 91 92 93 94 95 96 97
     * col          (col) int64 2 3 4 5 6 7 8 9 10 ... 89 90 91 92 93 94 95 96 97
   Attributes:
       offset_row_col:  2

The resize method of the common module, allows to restitute disparity maps and masks with the original size
: add the pixels that have been truncated:

.. sourcecode:: text

   <xarray.Dataset>
   Dimensions:      (col: 100, row: 100)
   Coordinates:
     * row          (row) int64 0 1 2 3 4 5 6 7 8 9 ... 91 92 93 94 95 96 97 98 99
     * col          (col) int64 0 1 2 3 4 5 6 7 8 9 ... 91 92 93 94 95 96 97 98 99
   Attributes:
       offset_row_col:  0

These pixels will have bit 0 set, *The point is invalid: left image edge*, in the :ref:`validity_mask` and
will be assigned the *border_disparity* ( configurable in the json configuration file , see :ref:`resize_parameters`  )
in the disparity maps.

Right image
^^^^^^^^^^^^^^^

Because of the disparity range choice, it is possible that there is no available point to scan on the right image.
In this case, matching cost cannot be computed for this pixel and the value will be set to :math:`nan` .
Then bit 1 will be set : *The point is invalid: the disparity interval to explore is
absent in the right image* and the point disparity will be set to *invalid_disparity*.
