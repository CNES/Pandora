.. _inputs:

Inputs
======

Pandora works with two stereo rectified one-channel or multi-channel images.


Configuration and parameters
****************************

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Required**
    *img_left*,Path to the left image,string,,Yes
    *img_right*,Path to the right image,string,,Yes
    *nodata_left*,Nodata value for left image, int,-9999,No
    *nodata_right*,Nodata value for right image,int,-9999,No
    *disp_min*,Minimal disparity,int or string,,Yes
    *disp_max*,Maximal disparity,int or string,,Yes
    *left_mask*,Path to the left mask,string,"none",No
    *right_mask*,Path to the right mask,string,"none",No
    *disp_min_right*,Path to the minimal disparity grid of the right image,string,"none",No
    *disp_max_right*,Path to the maximal disparity grid of the right image,string,"none",No
    *left_classif*,Path to the left classification map,string,"none",No
    *right_classif*,Path to the right classification map,string,"none",No
    *left_segm*,Path to the left segmentation map,string,"none",No
    *right_segm*,Path to the right segmentation map,string,"none",No

Images
******

- If the input images are multiband, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.
- Only one-band masks are accepted by pandora. Mask must comply with the following convention :
    - Value equal to 0 for valid pixel
    - Value not equal to 0 for invalid pixel
- For more details please see :ref:`as_an_api` images subsection


.. note::
    - Parameters *disp_min* and *disp_max* can be the disparity range (type int) or the path to the grids
      that contain the minimum and maximum disparity of a pixel (type string).
    - If *disp_min* and *disp_max* are integers, then the range of disparities is fixed. The minimal and maximal
      disparity of the right image are automatically calculated : *disp_min_right* = - *disp_max* and *disp_max_right* = - *disp_min*.
    - If *disp_min* or *disp_max* are strings, that means they are paths to grids of disparities which have the same size as the input images.
      Each pixel (x,y) of the grid corresponds to a local disparity (min for disp_min and max for disp_max) related to the same pixel (x, y) of the image.
    - Cross-checking step is not applicable if only left grids are provided (i.e the right one must be provided).

.. note::
    Mask must comply with the following convention
     - Value equal to 0 for valid pixel
     - Value not equal to 0 for invalid pixel

.. note::
    If the input images are multiband, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please
    see :ref:`faq`.

.. note::
    The input classification image must have one band per class (with value 1 on the pixels belonging to the class, and 0 for the rest), and the band's names must be present on the image metadata. To see how to add band's names on the classification image's metadata, please
    see :ref:`faq`.
