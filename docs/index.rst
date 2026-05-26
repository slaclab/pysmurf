Welcome to pysmurf's documentation!
===================================

The Python control software for `SMuRF <https://arxiv.org/abs/2208.10523>`_
(SLAC Microresonator RF). Provides low-level register commands and
high-level operations for tuning, tracking, and streaming data from
superconducting microresonator arrays.

.. note::

   This documentation corresponds to pysmurf |version|.
   Install from PyPI: ``pip install pysmurf-slac``

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   user/intro
   user/concepts
   user/installation
   user/configuration
   user/quickstart
   user/operations
   user/references
   user/developer


.. toctree::
   :maxdepth: 3
   :caption: Client API

   client/base
   client/command
   client/debug
   client/tune
   client/util

.. toctree::
   :maxdepth: 3
   :caption: Core API

   core/conventers
   core/counters
   core/devices
   core/emulators
   core/roots
   core/server_scripts
   core/transmitters
   core/utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
