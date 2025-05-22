.. _profiling:

Profiling
=========

Pandora has a built-in profiling tool, that can be used to provide more insight on the memory and time use of each step in the pipeline.

By default, two graphs will be created :

* an icicle graph, showing the time spent in each step of the pipeline
* a plot, showing the memory consumption of Pandora at regular intervals during the execution 


Configuration and parameters
****************************

Pandora's profiling configuration works just like a pipeline step, but is placed at the root of the config file : 

**profiling** 

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Required**
    *enabled*,Enable or disable the profiling,bool,False,No
    *save_graphs*,Save the graphs generated after displaying them,bool,False,No
    *save_raw_data*,Save the raw data on calls,bool,False,No
    *display_graphs*,Display the graphs automatically generated,bool,False,No

.. note::
    *profiling* can also be set to True or False directly instead of being a dict, setting every boolean inside its configuration.
   