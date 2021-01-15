.. _optimization:

Optimization
============

Theoretical basics
------------------

The third step is to minimize a global energy defined by

    :math:`E(d) = E_{data}(d) + \lambda E_{smooth}(d)`

First term, called data term represents raw matching cost measurement. The right one, the smoothness term, represents smoothness assumptions made
by the algorithm.

The methods available in Pandora are

- Semi-Global Matching [Hirschmuller2007]_, made available by :ref:`plugin_libsgm`.




Configuration and parameters
----------------------------

+-----------------------+----------------------+--------+---------------+-------------------------------------+----------+
| Name                  | Description          | Type   | Default value | Available value                     | Required |
+=======================+======================+========+===============+=====================================+==========+
| *optimization_method* | Optimization method  | string |               | "sgm" if plugin_libsgm is installed | Yes      |
+-----------------------+----------------------+--------+---------------+-------------------------------------+----------+

Optimization method:

- *sgm*: :ref:`plugin_libsgm_conf` of :ref:`plugin_libsgm` for sub-parameters and configuration example.

