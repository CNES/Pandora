<h1 align="center">
  <a href="https://pandora.readthedocs.io/?badge=latest"><img src="https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/logo/logo_typo_large.png?inline=false" alt="Pandora Stereo Framework" width="432"></a>
</h1>

<h4 align="center">A stereo matching framework that will help you design your stereo matching pipeline with state of the art performances.</h4>

<p align="center">
  <a><img src="https://github.com/CNES/Pandora/actions/workflows/pandora_ci.yml/badge.svg?branch=master"></a>
  <a href="https://codecov.io/gh/CNES/Pandora"><img src="https://codecov.io/gh/CNES/Pandora/branch/master/graph/badge.svg?token=IENWO02GB3"/></a>
  <a href='https://pandora.readthedocs.io/?badge=latest'><img src='https://readthedocs.org/projects/pandora/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://opensource.org/licenses/Apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://mybinder.org/v2/gh/CNES/Pandora/master"><img src="https://mybinder.org/badge_logo.svg"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#firststep">First Step</a> •
  <a href="#customize">Customize</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>


## Overview

From stereo rectified images to disparity map  |  Pandora is working with cost volumes
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/schema_readme.png?inline=false)  |  ![](https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/disparity3D_with_projected_dispartiry_color.gif)


Pandora aims at shortening the path between a stereo-matching prototype and its industrialized version.  
By providing a modular pipeline inspired from the (Scharstein et al., 2002) taxonomy, it allows one to emulate, analyse and hopefully improve state of the art stereo algorithms with a few lines of code. 

We (CNES) have actually been using Pandora to create the stereo matching pipeline for the CNES & Airbus <a href="https://co3d.cnes.fr/en/co3d-0"><img src="https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/logo_co3D_cnes.jpg" width="32" height="32"/></a> off board processing chain.  
Leaning on Pandora's versatility and a fast-paced constantly evolving field we are still calling this framework a work in progress !

<img src="https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/pandora_first_step_terminal.gif" width="500"/>

## Install

Pandora is available on Pypi and can be installed by:

```bash
pip install pandora
```

For stereo reconstruction we invite you to install pandora **and** the required plugins using instead the following shortcut:

```bash
pip install pandora[sgm, mccnn]
```

## First step

Pandora requires a `config.json` to declare the pipeline and the stereo pair of images to process. 
Download our data sample to start right away ! 
- [cones stereo pair](https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/cones.zip) 
- [a configuration file](https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_local_block_matching.json)

```bash
# install pandora latest release
pip install pandora

# download data samples
wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/cones.zip  # input stereo pair
wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_local_block_matching.json # configuration file

# uncompress data
unzip cones.zip

# run pandora
pandora a_local_block_matching.json output_dir

#Left (respectively right) disparity map is saved in output_dir/left_disparity.tif (respectively output_dir/right_disparity.tif)
```

## To go further

To create you own stereo matching pipeline and choose among the variety of algorithms we provide, please consult [our online documentation](https://pandora.readthedocs.io/index.html).

You will learn:
- which stereo matching steps you can [use and combine](https://pandora.readthedocs.io/userguide/step_by_step.html)
- how to quickly set up a [Pandora pipeline](https://pandora.readthedocs.io/userguide/sequencing.html)
- how to add your own private algorithms to [customize your Pandora Framework](https://pandora.readthedocs.io/developer_guide/your_plugin.html)
- how to use [Pandora API](https://pandora.readthedocs.io/userguide/as_an_api.html) (see [CARS](https://github.com/CNES/CARS) for real life exemple)

## Credits

Our data test sample is based on the 2003 Middleburry dataset (D. Scharstein & R. Szeliski, 2003).

*(D. Scharstein & R. Szeliski, 2002). Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. International journal of computer vision, 47(1-3), 7-42.*  
*(D. Scharstein & R. Szeliski, 2003). Scharstein, D., & Szeliski, R. (2003, June). High-accuracy stereo depth maps using structured light. In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings. (Vol. 1, pp. I-I). IEEE.*

## Related

[Plugin_LibSGM](https://github.com/CNES/pandora_plugin_libsgm) - Stereo Matching Algorithm plugin for Pandora  
[Plugin_MC-CNN](https://github.com/CNES/pandora_plugin_mccnn) - MC-CNN Neural Network plugin for Pandora  
[CARS](https://github.com/CNES/CARS) - CNES 3D reconstruction software

## References

Please cite the following paper when using Pandora:   
*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*




