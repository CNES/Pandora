.. _develop_plugin:

How to develop a plugin
=======================

One of the strengths of Pandora framework is the ability to develop an algorithm and use it as an external plugin.
This mechanism makes it possible to enrich methods avalaible for each step of Pandora.

The following sections describe the 4 major rules to respect in order to create your python package you want to use as
a Pandora plugin.

1. Instantiate a subclass

Each subpackage of Pandora, representing one particular step, defines an `abstract base classes <https://docs.python.org/3/library/abc.html>`_

.. sourcecode:: python

    @stereo.AbstractStereo.register_subclass('my_stereo_method')
    class MYSTEREO(stereo.AbstractStereo):

2. Provide definitions of abstract methods

.. sourcecode:: python

    def desc(self):
        """
        Describes the stereo method
        """
        print('My similarity measure')

    def compute_cost_volume(self, img_left, img_right, disp_min, disp_max, **cfg)
        """
        """
        print ("Just an example")

3. Implement a configuration checking method for parameters checking

.. sourcecode:: python

    def __init__(self, **cfg):

        self.cfg = self.check_config(**cfg)
        self._my_stereo_param = str(self.cfg['my_stereo_param'])
        self._window_size = self.cfg['window_size']
        self._subpix = self.cfg['subpix']

    def check_config(self, **cfg):
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: stereo configuration
        :type cfg: dict
        :return cfg: stereo configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'window_size' not in cfg:
            cfg['window_size'] = self._WINDOW_SIZE
        if 'subpix' not in cfg:
            cfg['subpix'] = self._SUBPIX

        schema = {
            "stereo_method": And(str, lambda x: x == 'my_stereo_method'),
            "window_size": And(int, lambda x: x == 11),
            "subpix": And(int, lambda x: x == 1),
            "my_stereo_param": int,
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg desc(self):

4. Make your plugin avalaible

Pandora works with `entry point specification <https://packaging.python.org/specifications/entry-points/>`_
and can load all plugin refered in the "pandora.plugin" group.

So, you must declare, on your setup.py file, an entry point:

.. sourcecode:: python

    setup(name='plugin_my_stereo_method',
          setup_requires=['very-good-setuptools-git-version'],
          description='Pandora plugin to compute cost volume with my new matching cost algorithm',
          long_description=readme(),
          packages=find_packages(),
          install_requires=requirements,
          entry_points="""
              [pandora.plugin]
              plugin_my_stereo_method = plugin_my_stereo_method.my_stereo_method:MYSTEREO
          """,
          cmdclass=cmdclass,
          )



