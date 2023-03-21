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
