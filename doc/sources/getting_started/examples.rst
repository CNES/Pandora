Examples
========

Command line
------------

Define your inputs and configure your pipeline by writting a json configuration file

.. sourcecode:: text

    {
      "input" : {
        "img_ref" : "tests/pandora/ref.png",
        "img_sec" : "tests/pandora/sec.png",
        "disp_min" : -60,
        "disp_max" : 0
      },
      "pipeline": {
        "stereo" : {
          "stereo_method": "zncc",
          "window_size": 5,
          "subpix": 4
      },
      "invalid_disparity": 0
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
        img_ref = read_img(cfg['input']['img_ref'], no_data=cfg['image']['nodata1'], cfg=cfg['image'],
                           mask=cfg['input']['ref_mask'])
        img_sec = read_img(cfg['input']['img_sec'], no_data=cfg['image']['nodata2'], cfg=cfg['image'],
                           mask=cfg['input']['sec_mask'])

        # Read range of disparities
        disp_min = read_disp(cfg['input']['disp_min'])
        disp_max = read_disp(cfg['input']['disp_max'])
        disp_min_sec = read_disp(cfg['input']['disp_min_sec'])
        disp_max_sec = read_disp(cfg['input']['disp_max_sec'])

        # Run the Pandora pipeline
        ref, sec = run(pandora_machine, img_ref, img_sec, disp_min, disp_max, cfg, disp_min_sec, disp_max_sec)

        # Save the reference and secondary DataArray in tiff files
        save_results(ref, sec, output)

        # Save the configuration
        save_config(output, cfg)
