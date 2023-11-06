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
    *disp_left*,"Path to the disparity grid of the left image or [min, max] values","string or [int, int]",,Yes
    *left_mask*,Path to the left mask,string,"none",No
    *right_mask*,Path to the right mask,string,"none",No
    *disp_right*,"Path to the disparity grid of the right image or [min, max] values","string or [int, int]","none",No
    *left_classif*,Path to the left classification map,string,"none",No
    *right_classif*,Path to the right classification map,string,"none",No
    *left_segm*,Path to the left segmentation map,string,"none",No
    *right_segm*,Path to the right segmentation map,string,"none",No

Images
******

- If the input images are multiband, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.
- For semantic segmentation classification, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.
- Only one-band masks are accepted by pandora. Mask must comply with the following convention :
    - Value equal to 0 for valid pixel
    - Value not equal to 0 for invalid pixel
- For more details please see :ref:`as_an_api` images subsection


.. note::
    - Parameter *disp_left* can be the disparity range (type list[int, int]) or the path to the grids
      that contain the minimum and maximum disparity of a pixel (type string).
    - If *disp_left* is a tuple of integers, then the range of disparities is fixed. The minimal and maximal
      disparity of the right image are automatically calculated :
      *disp_right[0]* = - *disp_left[1]* and *disp_right[1]* = - *disp_left[0]*
      where index `0` correspond to *min* and index `1` correspond to *max*.
    - If *disp_left* is a string, that means it is the path to grids of disparities which have the same size as the input images.
      Each pixel (x,y) of the grid corresponds to a local disparity (min for disp_left[0] and max for disp_left[1]) related to the same pixel (x, y) of the image.
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
