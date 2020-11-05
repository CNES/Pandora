# Changelog

## Unreleased

### Fixed

- Correct stereo methods in case the images have nan values. [#150]

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
