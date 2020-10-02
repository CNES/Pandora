Examples
========

Command line
------------

Define your inputs and configure your pipeline by writting a json configuration file

.. sourcecode:: text

    {
      "input" : {
        "img_left" : "tests/pandora/left.png",
        "img_right" : "tests/pandora/right.png",
        "disp_min" : -60,
        "disp_max" : 0
      },
      "pipeline": {
          "stereo": {
            "stereo_method": "ssd",
            "window_size": 5,
            "subpix": 1
          },
          "disparity": {
            "disparity_method": "wta",
            "invalid_disparity": "np.nan"
          },
          "filter": {
            "filter_method": "median"
          }
          "resize": {
            "border_disparity": "np.nan"
          }
      }
    }

And run pandora

.. sourcecode:: text

    pandora config.json output_dir


As a package
------------

.. sourcecode:: python

    import pandora
    from pandora.JSON_checker import check_conf, read_config_file
    from pandora.img_tools import read_img
    from pandora.common import save_results, save_config
    from pandora.constants import *
    from pandora import import_plugin, check_conf
    from .state_machine import PandoraMachine


    def pandora_stereo(cfg_path, output, verbose):
        """
        Check config file and run pandora framework

        :param cfg_path: path to the json configuration file
        :type cfg_path: string
        :param output: Path to output directory
        :type output: string
        :param verbose: verbose mode
        :type verbose: bool
        """
        user_cfg = read_config_file(cfg_path)

        # Import pandora plugins
        import_plugin()

        # Instantiate pandora state machine
        pandora_machine = PandoraMachine()

        # check the configuration
        cfg = check_conf(user_cfg, pandora_machine)

        # Read images and masks
        img_left = read_img(cfg['input']['img_left'], no_data=cfg['image']['nodata1'], cfg=cfg['image'],
                           mask=cfg['input']['left_mask'])
        img_right = read_img(cfg['input']['img_right'], no_data=cfg['image']['nodata2'], cfg=cfg['image'],
                           mask=cfg['input']['right_mask'])

        # Read range of disparities
        disp_min = read_disp(cfg['input']['disp_min'])
        disp_max = read_disp(cfg['input']['disp_max'])
        disp_min_right = read_disp(cfg['input']['disp_min_right'])
        disp_max_right = read_disp(cfg['input']['disp_max_right'])

        # Run the Pandora pipeline
        left, right = run(pandora_machine, img_left, img_right, disp_min, disp_max, cfg, disp_min_right, disp_max_right)

        # Save the left and right DataArray in tiff files
        save_results(left, right, output)

        # Save the configuration
        save_config(output, cfg)
