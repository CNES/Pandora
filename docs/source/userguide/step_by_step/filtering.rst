.. _filter:

Filtering of the disparity map
==============================

Theoretical basics
------------------

The filtering methods allow to homogenize the disparity maps, those available in pandora are :

- median filter.
- bilateral filter.

.. note::  Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is ignored for the calculation


Configuration and parameters
----------------------------

+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| Name            | Description                | Type   | Default value | Available value       | Required                           |
+=================+============================+========+===============+=======================+====================================+
| *filter_method* | Filtering method           | string |               | "median", "bilateral" | Yes                                |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *filter_size*   | Filter's size              | int    | 3             | >= 1                  | No                                 |
|                 |                            |        |               |                       | Only avalaible if median filter    |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *sigma_color*   | Bilateral filter parameter | float  | 2.0           |                       | No                                 |
|                 |                            |        |               |                       | Only avalaible if bilateral filter |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+
| *sigma_space*   | Bilateral filter parameter | float  | 6.0           |                       | No                                 |
|                 |                            |        |               |                       |                                    |
|                 |                            |        |               |                       | Only avalaible if bilateral filter |
+-----------------+----------------------------+--------+---------------+-----------------------+------------------------------------+


**Example**

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
            "filter":
            {
                "filter_method": "median"
            }
            ...
        }
    }