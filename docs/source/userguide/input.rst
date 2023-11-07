.. _inputs:

Inputs
======

Pandora works with two stereo rectified one-channel or multi-channel images.


Configuration and parameters
****************************

Pandora input configuration files are divided into two parts: left and right. 

**Left input** 

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Required**
    *img*,Path to the left image,string,,Yes
    *nodata*,Nodata value for left image, int,-9999,No
    *disp*,"Path to the disparity grid of the left image or [min, max] values","string or [int, int]",,Yes
    *mask*,Path to the left mask,string,"none",No
    *classif*,Path to the left classification map,string,"none",No
    *segm*,Path to the left segmentation map,string,"none",No

**Right input** 

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Required**
    *img*,Path to the right image,string,,Yes
    *nodata*,Nodata value for right image,int,-9999,No
    *disp*,"Path to the disparity grid of the right image or [min, max] values","string or [int, int]","none",No
    *mask*,Path to the right mask,string,"none",No
    *classif*,Path to the right classification map,string,"none",No
    *segm*,Path to the right segmentation map,string,"none",No

**Requirement** 

.. note::
    *disp* parameter is only required for left image. 
   
Images
******

- If the input images are multiband, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.
- For semantic segmentation classification, the band's names must be present on the image metadata. To see how to add band's names on the image's metadata, please see :ref:`faq`.
- Only one-band masks are accepted by pandora. Mask must comply with the following convention :
    - Value equal to 0 for valid pixel
    - Value not equal to 0 for invalid pixel
- For more details please see :ref:`as_an_api` images subsection


.. note::
    - Parameter left *disp* can be the disparity range (type list[int, int]) or the path to the grids
      that contain the minimum and maximum disparity of a pixel (type string).
    - If left *disp* is a tuple of integers, then the range of disparities is fixed. The minimal and maximal
      disparity of the right image are automatically calculated :
      right *disp[0]* = - left *disp[1]* and right *disp[1]* = - left *disp[0]*
      where index `0` correspond to *min* and index `1` correspond to *max*.
    - If left *disp* is a string, that means it is the path to grids of disparities which have the same size as the input images.
      Each pixel (x,y) of the grid corresponds to a local disparity (min for left *disp[0]* and max for left *disp[1]*) related to the same pixel (x, y) of the image.
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
