# Changelog

## Unreleased

## 1.2.1 (December 2021)

### Fixed
- Functions get_margins works now with missing parameters (refinement, filter) [#235]
- Enable inf value as a nodata. [#238]
- Modify exception handling with machine state. [#239]

### Added

- Add band names on confidence measure output. [#234]

### Changed

- Update python packaging. [#197]

## 1.2.0 (September 2021)

- Add a risk measure. [#228]
- Subpixel processing with any even value. [#225]

### Fixed

- Correction on sampled_ambiguity when all costs are NaN. [#230]

## 1.1.1 (July 2021)

### Fixed

- Correct refinement bug when using subpixel. [#223]
- Corrections to bug after new xarray version. [#224]

### Added

- Memory consumption estimation. [#222]
- Add extra requirements making mccnn plugin available.

## 1.1.0 (June 2021)

### Added

- Binder for Pandora's notebooks. [#215] 
- Version handling with setuptools_scm. [#212]
- Set dataset's transform to None if it is the identity matrix. [#211]
- Handling of np.inf values on input images. [#210]
- Remove copy from check_dataset and doesn't return dataset anymore. [#219]
- Doc for confidence and piecewise optimization.
- Do not force omp layer on numba and silence its warning.

### Fixed

- Correction on multi scale processing to allocate the confidence measure. [#217]
- Correct non-masked nan values on validity mask. [#218]
- Margin computation without optimisation step. [#220]

## 1.0.0 (March 2021)

### Added

- Implementation of the ambiguity confidence measure. [#162] 
- Implementation of the  bilateral filter to suppress OpenCV dependency. [#148] 
- Checker to check if an image dataset is valid. [#196]
- Creation of Jupyter notebooks for Pandora.[#153]

### Fixed

- Correct Nan propagation of the bilateral filter. [#159]

### BREAKING CHANGE

- GeoRef is now supported and both `crs` & `transform` ought to be set as attributes of images dataset provided to Pandora [#158]

## 0.5.1 (January 2021)

### Fixed

- Call to interpolation functions. [#175]
- Check right disparities. [#177]
- Disparity refinement after disparity map filtering step. [#154]

## 0.5.0 (January 2021)

### Added 

- Implementation of multi scale processing for the disparity map computation. [#80]
- A `./data_sample/` directory with images and configuration files for user's first steps with Pandora [#168]
- Add extra requirements making sgm plugin available [#183]

### Changed

- Semantic change: stereo becomes matching_cost (transition name, module, files and class name). [#170]
- Merge image on input configuration section. [#169]
- Enable the use of GraphMachine if graphviz avalaible to generate machine states diagram. [#149]
- Move find_valid_neighbors function to img_tools. [#184]
- Renamed json_cheker to check_json. [#182]

### Fixed

- Confidence measure that computes the distance LR / RL in cross checking method. [#155]


## 0.4.0 (November 2020)

### Added

- Classification and segmentation maps as input. [#126]

### Changed

- Non truncation of the cost volume and the disparity map, which have the input image's size. [#152]
- Input masks convention and creation of masks dataset.

## 0.3.0 (October 2020)

### Added

- Implementation of a state machine for Pandora's sequencing management. [#107]
- Disparity grids as input to define dmin and dmax values for each pixel. [#91]
- A changelog file. [#115]

### Changed

- "Image configuration (no data of image, no data and invalid pixel of input mask ) is now
  implemented as attributes of xarray image dataset and must no longer be provided separately to the run method. [#131]
- The reference/secondary naming convention becomes left/right naming. [#135]
- Functions which can modify input arguments passed by reference does not return these references anymore. [#109]

### Fixed

- Creation of mask dataset in `read_img` function of the `img_tools` module. [#133]
- Management of no data in cbca method. [#125]
- Detection between false matches and occlusions. [#132]
