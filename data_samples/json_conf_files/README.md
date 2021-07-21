The following configuration files are provided as data samples:
- `a_local_block_matching.json` file with a `subpix` value of `4` to avoid aliasing artefacts with `zncc` similarity measure
- `a_semi_global_matching.json` file with a full pipeline (filter, validation steps also included -with actually two filters because a) why not; b) it is possible.-)
  - :warning: make sure the `pandora_plugin_libsgm` has been pip installed
    - `pip install pandora_plugin_libsgm` to install the plugin
    - or `pip install pandora[sgm]` 
- `a_semi_global_matching_with_mccnn_similarity_measure.json` file with a full pipeline using mccnn similarity measure 
  - :warning: make sure the `pandora_plugin_mccnn` has been pip installed
    - `pip install pandora_plugin_mccnn` to install the plugin
    - or `pip install pandora[mccnn]` 
