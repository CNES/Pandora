# Pandora

This CNES software implements a stereo pipeline which produces disparity map from a pair of images. Several state-of-the-art 
stereo algorithms are available. The implementation  is  modular  and  can  easily  be  ex-tended to include other 
algorithms. 

The pipeline performs the following steps:

    1. matching cost computation
    2. cost (support) aggregation
    3. optimization
    3. disparity computation
    4. subpixel cost refinement
    5. disparity filter, validation
    6. multiscale

## Dependencies

Dependencies are :

    python >= 3.6, pip, nose2, xarray >= 0.13.*, numpy, scipy, numba >= 0.47.*, rasterio, opencv-python,
    json-checker

## Installation

```bash
    pip install pandora
``` 

**From sources**

```bash
    git clone https://github.com/CNES/Pandora.git
    cd Pandora
    pip install .
```

**Build documentation**

Make sure  latex and dvipng are already installed

```
pip install sphinx-rtd-theme
pip install sphinx-autoapi
python setup.py build_sphinx
```

Documentation is built in pandora/doc/build/html 

## Usage

Run the python script `pandora` with a json configuration file (see `conf` folder as an example) and the following arguments :

```bash
    usage: pandora [-h] [-v] config

    Pandora stereo matching
    
    positional arguments:
      output_dir          Path to the output director
      config              Path to a json file containing the input files paths and the algorithm parameters
    
    optional arguments:
      -h, --help          show this help message and exit
      -v, --verbose       Increase output verbosity
```

The config file `config.json` is formatted as :

    {
        "input" : {
            "img_left" : "PATH/TO/img_left.tif",
            "img_right" : "PATH/TO/img_right.tif",
            "disp_min" : -100,
            "disp_max" : 100,
            "left_mask" : "PATH/TO/left_mask.tif",
            "right_mask" : "PATH/TO/right_mask.tif"
        }
    }

Mandatory fields are :
   - `img_left` : Path to the left image
   - `img_right` : Path to the right image
   - `disp_min` : Minimal disparity
   - `disp_max` : Maximal disparity


Optional fields are :
   - `left_mask` : Path to the left mask
   - `right_mask` : Path to the right mask
   - `left_classif` : Path to the left classification map
   - `right_classif` : Path to the right classification map
   - `left_segm` : Path to the left segmentation map
   - `right_segm` : Path to the right segmentation map



Pandora can also be used as a package : 

    import pandora

Input stereo images must be grayscale images. Parameters `nodata1` and `nodata2` of the json configuration file allow to specify
the value of no data in the left and right images.

The masks are optional, and have the following convention: 0 is a valid pixel, everything else is considered as masked.

All the parameters of the algorithm are stored in the json file. Configuration examples are available in the `conf` folder.
The file `pandora/config.py` lists all possible parameters and provides some explanations on the role of these parameters. 
The following paragraphs describe the methods currently implemented for each step of the pipeline.

#### Matching cost computation

The first step calculates a cost volume from the two images and one of the following similarity measures:

    - Sum of absolute differences
    - Sum of squared differences
    - Census
    - Zero mean normalized cross correlation

#### Cost (support) aggregation

Aggregate the matching cost by summing or averaging over a support region in the cost volume:

    - Cross-based Cost Aggregation

#### Optimization

Find a disparity function that minimizes a global energy in the cost volume:

-   Semi-Global Matching, by using [pugin_libsgm](https://github.com/CNES/Pandora_plugin_libsgm.git) and [libsgm](https://github.com/CNES/Pandora_libsgm.git)

SGM reference: [H. Hirschmuller. Stereo processing by semiglobal matching and mutual information. 
IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341]

#### Disparity computation

Choose at each pixel the disparity associated with the minimum cost:

    - Winner-take-all

#### Subpixel cost refinement

The sub-pixel refinement of disparities:

    - vfit
    - quadratic

#### Disparity filter, validation

Post-processing the computed disparities:

    - median filter
    - bilateral filter
    - cross checking     
    - mismatches and occlusions detection

#### Multiscale processing

Doing the whole pipeline for each number of scales

    - image pyramid computation
    - processing from coarse to fine
    - disparity range refinement with the results from the previous coarser scale
    
## Notes

For tests, we use images coming from 2003 Middleburry dataset 
(D. Scharstein and R. Szeliski. High-accuracy stereo depth maps using structured light.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2003), 
volume 1, pages 195-202, Madison, WI, June 2003.)


## References

If you use this CNES software, please cite the following paper: 

Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. 
Ground-truth generation and disparity estimation for optical satellite imagery.
ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.

