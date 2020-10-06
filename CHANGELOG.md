# Changelog

## Unreleased

## 0.3.0 (October 2020)

### Added

- Implementation of a state machine for Pandora's sequencing management. 
- Disparity grids as input to define dmin and dmax values for each pixel.

### Changed

- Image configuration (no data of image, no data and invalid pixel of input mask ) as attributes of  xarray image dataset.
- The reference/secondary naming convention becomes left/right naming.
- Functions which can modify input arguments passed by reference does not return these references anymore.
- Use of rasterio instead of gdal for reading and writing operations.

### Fixed

- Creation of mask dataset in `read_img` function of the `img_tools` module.
- Management of no data in cbca method.
- Detection between false matches and occlusions.

