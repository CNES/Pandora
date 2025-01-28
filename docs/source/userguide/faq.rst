.. _faq:

FAQ
=========

How do I add the band's names as metadata on a multiband image ?
****************************************************************

**Example using a python script and rasterio:**

.. code-block:: bash

    import rasterio
    from rasterio.enums import ColorInterp

    img_ds = rasterio.open("ORIGINAL_MULTIBAND_IMAGE_PATH")
    # Read the original image with rasterio
    # In this case, we consider that the input is an RGB image
    # with three bands, without any band metadata
    data_array = img_ds.read()
    nb_band, nb_row, nb_col = data_array.shape
    with rasterio.open(
        "FINAL_MULTIBAND_IMAGE_PATH",
        mode="w+",
        driver="GTiff",
        width=nb_col,
        height=nb_row,
        count=nb_band,
        dtype=rasterio.dtypes.float32,
        crs=img_ds.crs,
        transform=img_ds.transform,
    ) as source_ds:
        # Optional, color interpreter may be added to the band metadata
        source_ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
        # Band descriptions. This is the information that pandora will read
        descriptions = ["r", "g", "b"]
        for band in range(0, nb_band):
            # Band indexing starts at 1
            source_ds.write_band(band+1, data_array[band, :, :])
            source_ds.set_band_description(band+1, descriptions[band])

.. code-block::

    gdalinfo left_rgb.tif

    Driver: GTiff/GeoTIFF
    Files: left_rgb.tif
    Size is 450, 375
    Image Structure Metadata:
      INTERLEAVE=PIXEL
    Corner Coordinates:
    Upper Left  (    0.0,    0.0)
    Lower Left  (    0.0,  375.0)
    Upper Right (  450.0,    0.0)
    Lower Right (  450.0,  375.0)
    Center      (  225.0,  187.5)
    Band 1 Block=450x1 Type=Float32, ColorInterp=Red
      Description = r
    Band 2 Block=450x1 Type=Float32, ColorInterp=Green
      Description = g
    Band 3 Block=450x1 Type=Float32, ColorInterp=Blue
      Description = b


How do I check my data without launching Pandora completely ?
*************************************************************


**Example using a python script and pandora library:**

User configuration file, *pandora_conf.json*:

.. code:: json
    :name: user configuration example

    {
      "input": {
        "left": {
          "img": "./left_rgb.tif",
          "disp": [-60, 0]
        },
        "right": {
          "img": "./right_rgb.tif",
        }
      },
      "pipeline": {
        "matching_cost": {
          "matching_cost_method": "zncc",
          "band": "r",
          "window_size": 5,
          "subpix": 4
        },
        "disparity": {
          "disparity_method": "wta",
          "invalid_disparity": "NaN"
        },
        "refinement": {
          "refinement_method": "quadratic"
        },
        "validation" : {
          "validation_method": "cross_checking_accurate"
        }
      }
    }


And the python script.

.. code-block:: bash

    from pandora.img_tools import create_dataset_from_inputs
    from pandora.check_configuration import check_dataset, read_config_file

    # Read pandora_conf.json
    user_cfg = read_config_file(cfg_path)

    # Read images 
    img_left = create_dataset_from_inputs(input_config=cfg['input']["left"])
    img_right = create_dataset_from_inputs(input_config=cfg['input']["right"])

    # Check datasets: shape, format and content
    check_datasets(img_left, img_right)
