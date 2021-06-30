As an API
=========

Pandora API usage
*****************

Pandora provides a full python API which can be used to compute disparity maps as show in this basic example:

.. sourcecode:: python

    import pandora
    from pandora import common
    from pandora.img_tools import read_img, read_disp
    from pandora.check_json import check_conf, read_config_file
    from pandora.state_machine import PandoraMachine

    def pandora_stereo(cfg_path: str, output: str, verbose: bool) -> None:
        """
        Check config file and run pandora framework accordingly

        :param cfg_path: path to the json configuration file
        :type cfg_path: string
        :param output: Path to output directory
        :type output: string
        :param verbose: verbose mode
        :type verbose: bool
        :return: None
        """

        # Read the user configuration file
        user_cfg = read_config_file(cfg_path)

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate pandora state machine
        pandora_machine = PandoraMachine()
        # check the configuration
        cfg = check_conf(user_cfg, pandora_machine)

        # setup the logging configuration
        pandora.setup_logging(verbose)

        # Read images and masks
        img_left = read_img(cfg['input']['img_left'], no_data=cfg['input']['nodata_left'],mask=cfg['input']['left_mask'],
                            classif=cfg['input']['left_classif'], segm=cfg['input']['left_segm'])
        img_right = read_img(cfg['input']['img_right'], no_data=cfg['input']['nodata_right'],
                             mask=cfg['input']['right_mask'], classif=cfg['input']['right_classif'],
                             segm=cfg['input']['right_segm'])

        # Read range of disparities
        disp_min = read_disp(cfg['input']['disp_min'])
        disp_max = read_disp(cfg['input']['disp_max'])
        disp_min_right = read_disp(cfg['input']['disp_min_right'])
        disp_max_right = read_disp(cfg['input']['disp_max_right'])

        # Run the Pandora pipeline
        left, right = pandora.run(pandora_machine, img_left, img_right, disp_min, disp_max, cfg['pipeline'], disp_min_right,
                          disp_max_right)

        # Save the left and right DataArray in tiff files
        common.save_results(left, right, output)

        # Save the configuration
        common.save_config(output, cfg)

        if __name__ == '__main__':

            pandora_stereo('./data_samples/json_conf_files/a_local_block_matching.json', './output/', True)

If you want to learn  more, please consult our `Pandora Api tutorial notebook <https://github.com/CNES/Pandora/tree/master/notebooks/...>`_.
It will help you to understand, manipulate and customize our API.

Pandora's data
**************

Images
######

Pandora reads input images before stereo computation and creates two datasets for left and right
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
        crs:           None
        transform:     | 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, 0.00, 1.00|
        valid_pixels:  0
        no_data_mask:  1

Two data variables are created in this dataset:

 * *im*: contains input image data
 * *msk*: contains input mask data + no_data of input image

.. note::
    This example comes from a dataset created by Pandora's reading function. Dataset attributes
    *valid_pixels* and *no_data_mask* cannot be modified with this function. Its indicate the *msk*
    data convention.
    For API user who wants to create own dataset, without using Pandora's reading function, it is
    possible to declare its own mask convention with these attributes:

      * *no_data_img* : value of no_data in input image
      * *valid_pixels*: value of valid pixels in input mask
      * *no_data_mask*: value of no_data pixel in input mask

Cost volume
###########

Pandora generates a cost volume during the first step: *Matching cost computation*. The cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.
When matching is impossible, the matching cost is np.nan.

This Dataset also has a :

- xarray.DataArray 3D confidence_measure, which contains quality indicators, depending on what is activated. It can be enriched by indicators calculated in the different plugins.
- xarray.DataArray disp_indices, which contains the minimum cost indices calculated in step *Disparity computation*.


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
        crs:             None
        transform:       | 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, 0.00, 1.00|

The cost volume corresponds to the variable cv ( and cv_right for the right / left cost volume ) in the file pandora/__init__.py :

.. note::

    The cost volume contains only the similarity factors calculated with the steps *Calculation of mapping costs*,
    *Aggregation of costs*, *Optimization*. It does not contain the interpolated factors ( calculated in step
    *disparity refinement*), these are available in the *interpolated_coeff* variable in the Disparity Dataset.

Disparity map
#############

The *Disparity computation* step generates a disparity map in cost volume geometry. This disparity map is
a float32 type 2D xarray.DataArray, stored in a xarray.Dataset.
This Dataset also has a :

- xarray.DataArray 3D confidence_measure, which contains quality indicators, depending on what is activated. It can be enriched by indicators calculated in the different plugins.
- xarray.DataArray validity_mask which represents the :ref:`validity_mask`.
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
        crs:                    None
        transform:              | 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, 0.00, 1.00|


Validity mask
#############

Validity masks are 2D xarray.DataArray and are 16-bit encoded: each bit represents a
rejection criterion (= 1 if rejection, = 0 otherwise): See :ref:`validity_mask`.

The validity masks are stored in the xarray.Dataset left and right in the pandora/__init__.py file.

.. _border_management:

Border management
#################

Left image
----------

Pixels of the left image for which the measurement thumbnail protrudes from the left image are set to :math:`nan` on the cost volume
For a similarity measurement with a 5x5 window, these incalculable pixels in the left image correspond
to a 2-pixel crown at the top, bottom, right and left, and are represented by the offset_row_col attribute in
the xarray.Dataset.

These pixels will have bit 0 set, *The point is invalid: left image edge*, in the :ref:`validity_mask` and
will be assigned the *invalid_disparity* ( configurable in the json configuration file ) in the disparity maps.

Right image
-----------

Because of the disparity range choice, it is possible that there is no available point to scan on the right image.
In this case, matching cost cannot be computed for this pixel and the value will be set to :math:`nan` .
Then bit 1 will be set : *The point is invalid: the disparity interval to explore is
absent in the right image* and the point disparity will be set to *invalid_disparity*.