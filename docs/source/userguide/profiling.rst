.. _profiling:

Profiling
=========

Pandora has a built-in profiling tool, that can be used to provide more insight on the memory and time use of each step in the pipeline.

.. warning::

    Make sure to install Pandora with development dependencies included.

By default, two graphs will be created :

* an icicle graph, showing the time spent in each step of the pipeline
* a plot, showing the memory consumption of Pandora at regular intervals during the execution 


Configuration and parameters
****************************

Pandora's profiling configuration works just like a pipeline step, but is placed at the root of the config file : 

**profiling** 

.. csv-table::

    **Name**,**Description**,**Type**,**Default value**,**Required**
    *save_graphs*,Save the default graphs generated,bool,False,No
    *save_raw_data*,Save the raw data on calls as a .pickle file,bool,False,No

.. note::
    *profiling* can also be set to True or False directly instead of being a dict, setting every boolean inside its configuration to the specified value.
   

Example configuration :

.. code-block::

    {
        "profiling": {
            "save_raw_data": true,
            "save_graphs": true
        },
        "input": {
            ...
        },
        "pipeline": {
            ...
        }
    }


Saved profiling data
********************

When *save_raw_data* is enabled, Pandora saves the profiling information as a .pickle file containing a pandas DataFrame with the following structure :

.. csv-table::

    **Name**,**Description**
    *level*,Depth of the function call in the profiling stack
    *parent*,UUID of the "parent" call (call that was running when this call was made)
    *name*,Understandable name given to the function call
    *uuid*,Unique identifier of the function call
    *time*,Time (in seconds) it took to execute the function
    *call_time*,Timestamp (in seconds) at which the call was made
    *memory*,Either None or a list of (timestamp memory) tuples representing memory consumption (in megabytes) at each timestamp during the function execution


Modifying the profiled functions
********************************

To include a function in the icicle time graph, simply add the *@profile* decorator to the function, providing a descriptive name.

If you also want to track memory usage over time for a specific function call, set *memprof=True* in the decorator.
If the function is too fast (or slow) for the default memory sampling interval, you can modify it with *interval* (in seconds).

.. code-block:: Python

    from pandora.profiler import profile

    @profile("my profiled function", memprof=True, interval=0.5)
    def my_function():
        ...
