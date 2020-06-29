Quick overview
==============

Pandora is a new stereo-matching framework, inspired by the work of (Scharstein, Szeliski, 2002). To estimate a disparity
map from two stereo-rectified images, Pandora provides the following steps: matching cost computation, cost aggregation,
cost optimization, disparity computation, subpixel disparity re-finement, disparity filtering and validation.

Pandora is easy to configure and provides some of the well-know algorithms for each step abovementioned.

Moreover, it is easy for one to develop his own algorithm for any steps of Pandora and to use it as an external plugin.

To use it, simply run :

.. sourcecode:: text

    pandora config.json output_dir

where `config.json` is a json file containing input files paths and the algorithm parameters and `output_dir` is the folder
to save output results.

Pandora can also be used as a python package.

.. sourcecode:: python

    import pandora

