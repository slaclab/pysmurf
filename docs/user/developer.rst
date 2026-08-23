Developer Guide
===============

Docstrings
----------

pysmurf uses NumPy-style docstrings rendered with Sphinx napoleon.
See the `numpydoc guide
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ for format
reference.

Building Docs
-------------

.. code-block:: bash

   cd docs/
   pip install -r requirements.txt
   make html
   # open _build/html/index.html

Architecture
------------

``SmurfControl`` is composed of mixins:

- ``SmurfCommandMixin`` -- low-level register access (``_caput``/``_caget``)
- ``SmurfTuneMixin`` -- resonator finding, eta scans, tracking setup
- ``SmurfUtilMixin`` -- streaming, data I/O, channel management
- ``SmurfNoiseMixin`` -- noise PSD measurement
- ``SmurfIVMixin`` -- IV curve acquisition
- ``SmurfConfigPropertiesMixin`` -- config file properties
- ``SmurfAtcaMonitorMixin`` -- ATCA hardware monitoring

Source layout::

   python/pysmurf/
   +-- client/
   |   +-- base/          SmurfControl, SmurfConfig, logging
   |   +-- command/       register-level commands (7000+ lines)
   |   +-- tune/          find_freq, setup_notches, eta_scan, tracking
   |   +-- util/          streaming, data reading, channel ops
   |   +-- debug/         IV curves, noise analysis
   +-- core/
       +-- roots/         hardware platform entry points
       +-- devices/       pyrogue device wrappers
       +-- transmitters/  custom data output

Testing
-------

.. code-block:: bash

   pytest python/pysmurf/client/test/
