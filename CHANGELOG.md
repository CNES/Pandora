# Changelog

## 1.6.3 (February 2025)

### Changed

- update pandora plugin libsgm version to 1.5.4

### Fixed

- normalisation of ambiguity: doc and bug
- fix meson.build errors and clean
- meson tests running
- add pybind11 dependency in makefile

## 1.6.3a1 (February 2025)

### Added

- Multi wheel cibuildwheel in github CI for release
- get disp min and disp max in datasets

### Changed

- Numba function refactored to C++ functions.
- Census refactored to c++.
- Lower and upper bounds from risk added.
- Clean and simplify README

### Fixed

- Confidence measure robust to zncc matching cost measure.
- properly manage multiband inputs in census
- update the get_coordinates method to static method


## 1.6.2 (September 2024)

## 1.6.2a1 (September 2024)

### Added

- Using the numba cache. [#413]
- Modification of check_conf to compare the disparity interval with the image size. [#411]
- Addition of a spline order parameter to the shift_right_img method. [#420]
- Add possibility to not use numba cache. [#421]

### Changed

- Update criteria documentation. [#410]
- Changing the after callback to before in the state machine. [#407]
- Updating the filtering.rst, a documentation file. [#414]
- Remove numpy warning. [#408]

### Fixed

- Solving problems displaying images in notebooks. [#409]
- Move the _indicator parameter to remove the sphinx warning. [#416]
- Risk normalisation by external global interval. [#418]
- Use of ‘max’ measurement type with ambiguity. [#404]

## 1.6.1 (June 2024)

### Changed

- Fix numpy version.

## 1.6.1a2 (June 2024)

### Fixed

- Fixed calculation of ambiguity and risk with variable disparity. [#400]

### Changed

- Ambiguity normalisation by external global interval. [#402]

## 1.6.1a1 (June 2024)

### Added

- Add criteria documentation in Exploring the Field section. [#380]
- Add a new method to add disparity grid in xarray dataset. [#389]
- Add get method to margins. [#394]
- Add interval_indicator on filetring documentation.

### Fixed

- Fix link to notebooks in the documentation. [#310]
- Update cv_masked to be used with a column step. [#391]
- Fix step in get_coordinates method.
- Fix disparities bigger than images shape. [#396]
- Update notebooks after evolution of pandora run_prepare method. [#395]

### Changed

- Uniformisation tests for cv_masked method. [#372]
- Extract col_to_compute from grid_estimation method. [#393]
- Update pandora after evolution of use_confidence in libsgm plugin. [#397]

## 1.6.0 (January 2024)

### Added

- Using new check_datasets function and modification to the check_band_names API. [#338]
- Added margin calculation for the treatment chain with ROI image. [#341]
- Addition of a method that estimates the grid to be calculated. [#342]
- Add calculations for criteria 0,1,2,6,7 to the previously allocated cost_volume. [#325]
- Add a new confidence mesure, interval_confidence.
- Added an additional check not to use sgm with a step other than 1. [#378]

### Fixed

- Fix memory_consumption_estimation method with disparity grids. [#367]
- Fix step in matching cost zncc method.
- Fix disparity_source word.
- Fix readthedocs. [#373]
- Fix warning for xarray.Dataset.dims. [#375]
- Fix margins for filter step. [#374]
- Fix error in compute_mean_raster method. [#379]
- Fix criteria computation order. [#382]
- Correction of mistakes in the documentation. [#381]

### Changed

- Update user configuration file with new keys : "left" & "right". [#314]
- Updating information in the various xarrays. [#368]
- Parametrization of numba parallelization.
- Replacing sys.exit by raises. [#370]
- Xarray coordinates updated if a ROI is used in the create_dataset_from_inputs function. [#371]
- Move allocate_cost_volume. [#343]
- Change pkg_resources to importlib.metadata. [#349]
- Update of the minimal version for python. [#377]
- Update of the minimal version for python for CI github. [#365]

## 1.6.0a1 (November 2023)

### Added

- Addition of a step for matching_cost (only usable with Pandora2d). [#337]
- Adding roi in create_dataset_from_inputs. [#346]
- Adding disparity in dataset image. [#331]
- Check inter dataset. [#334]
- Adding check_datasets function. [#335]

### Fixed

- Rationalisation of the allocate_costvolume function. [#320]
- Remove right_disp_map. [#324]
- Fix step use. [#353]
- Fix disparities for Pandora2d. [#359]
- Update get_metadata with classification and segmentation. [#361]
- Update check_disparities. [#363]
- Correction of notebooks. [#362]

### Changed

- Change dmin_dmax name function to get_min_max_from_grid. [#347]
- New disparity storage in the dataset containing the disparity map. [#339]
- Moving some "pipeline" checks to abstract classes. [#313]
- New format for disparity in the user configuration file. [#330]
- Change read_img function to create_dataset_from_inputs. [#345]
- Move get_metadata function in img_tools file. [#328]
- Update and move check_dataset function. [#333]

## 1.5.0 (April 2023)

### Added

- Documentation plugin_arnn [#261]
- Multiband input image classification. [#256]

### Fixed

- Deletion of the pip install codecov of githubAction CI [#305].

### Changed

- Reformatting check conf [#299]

## 1.4.0 (March 2023)

### Added

- Check that both segmentations/classifications are present if 3SGM and validation are intended. [#259]
- Multiband images compatibility. [#251]
- Add semantic_segmentation pipeline step. [#257]
- Add a control margin size. [#285]
- Allows the choice to normalize ambiguity. [#284]

### Fixed

- Correct Binder's notebooks dependency. [#292]
- Correct statistical_and_visual_analysis notebook test. [#287]

## 1.3.0 (January 2023)

### Added

- Add indicator to distinguish different cost volume confidence maps. [#243]
- Add Makefile. [#248]
- Add confidence method naming according to the selected confidence method. [#268]
- Add compatibility to plugin_libsgm's 3SGM method. [#255]

### Changed

- Remove dependance to ipyvolume. [#250]
- Adapt confidence method naming to snake case style. [#278]

### Fixed

- Remove warnings from transition. [#252]

## 1.2.1 (December 2021)

### Fixed

- Functions get_margins works now with missing parameters (refinement, filter) [#235]
- Enable inf value as a nodata. [#238]
- Modify exception handling with machine state. [#239]

### Added

- Add band names on confidence measure output. [#234]

### Changed

- Force python version >= 3.7. [#241]
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
