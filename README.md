<div align="center">
<a target="_blank" href="https://github.com/CNES/pandora">
<picture>
  <img
    src="https://raw.githubusercontent.com/CNES/Pandora/master/docs/source/Images/logo/logo_typo_large.png?inline=false""
    alt="Pandora"
    width="40%"
  />
</picture>
</a>

<h4> Pandora, a stereo matching framework</h4>

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Documentation](https://readthedocs.org/projects/pandora/badge/?version=latest)](https://pandora.readthedocs.io/)
[![Github Action](https://github.com/CNES/Pandora/actions/workflows/pandora_ci.yml/badge.svg?branch=master)](https://github.com/CNES/Pandora/actions)
[![Codecov](https://codecov.io/gh/CNES/Pandora/branch/master/graph/badge.svg?token=IENWO02GB3)](https://codecov.io/gh/CNES/Pandora)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/CNES/Pandora/master)

<p>
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#Documentation">Documentation</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>

</div>

## Overview

From stereo rectified images to disparity map  |  Pandora is working with cost volumes
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/CNES/Pandora/master/docs/source/Images/schema_readme.png?inline=false)  |  ![](https://raw.githubusercontent.com/CNES/Pandora/master/docs/source/Images/disparity3D_with_projected_dispartiry_color.gif)

Pandora is a stereo matching flexible framework made for research and production with state of the art performances:

- Inspired from the (Scharstein et al., 2002) modular taxonomy, it allows one to emulate, analyse and hopefully improve state of the art stereo algorithms with a few lines of code.
- For production purpose, Pandora have been created for the CNES & Airbus <a href="https://co3d.cnes.fr/en/co3d-0">CO3D project</a> processing chain, as [CARS](https://github.com/CNES/CARS) core stereo matching tool.

The tool is open for contributions, contact us to pandora AT cnes.fr !

## Install

Pandora is available on Pypi and can be installed by:

```bash
pip install pandora
```

For stereo reconstruction, install pandora **with** following plugins:

```bash
# SGM regularization
pip install pandora[sgm]
#  MCCNN AI matching cost capability (heavy!)
pip install pandora[mccnn]
```

## Quick Start

```bash

# Download configuration file
wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_local_block_matching.json

# Download data samples
wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/cones.zip

# Uncompress data
unzip cones.zip

# Run pandora
pandora a_local_block_matching.json output_dir

# Left and right disparity maps are saved in output_dir: left_disparity.tif and right_disparity.tif
```

## Documentation

To go further, please consult [our online documentation](https://pandora.readthedocs.io/).

## Credits

- *Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. International journal of computer vision, 47(1-3), 7-42.*  
- *Scharstein, D., & Szeliski, R. (2003, June). High-accuracy stereo depth maps using structured light. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings. (Vol. 1, pp. I-I).*
- *2003 Middleburry dataset (D. Scharstein & R. Szeliski, 2003).*

## Related

[Plugin_LibSGM](https://github.com/CNES/pandora_plugin_libsgm) - Stereo Matching Algorithm plugin for Pandora  
[Plugin_MC-CNN](https://github.com/CNES/pandora_plugin_mccnn) - MC-CNN Neural Network plugin for Pandora  
[Pandora2D](https://github.com/CNES/Pandora2D) - CNES Image Registration framework based on Pandora, with 2D disparity maps.
[CARS](https://github.com/CNES/CARS) - CNES 3D reconstruction software

## References

Please cite the following papers when using Pandora:

- *Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*
- *Youssefi D., Michel, J., Sarrazin, E., Buffe, F., Cournet, M., Delvit, J., L’Helguen, C., Melet, O., Emilien, A., Bosman, J., 2020. Cars: A photogrammetry pipeline using dask graphs to construct a global 3d model. IGARSS - IEEE International Geoscience and Remote Sensing Symposium.*
