Configuration
=============

pysmurf uses a JSON configuration file per physical hardware setup.

Config File Location
--------------------

Stored in ``cfg_files/<site>/`` in the pysmurf repository. Start from
``cfg_files/template/template.cfg``.

Setup Steps
-----------

1. Create a directory for your site under ``cfg_files/``
2. Copy ``template.cfg`` into it
3. Set amplifier bias values, ``bias_line_resistance``, ``R_sh``
4. After first connection, run ``S.estimate_phase_delay(band)`` and
   save results to the config

Usage
-----

.. code-block:: python

   S = pysmurf.SmurfControl(cfg_file='/path/to/my_config.cfg')

Band Parameters (``init.band_#``)
---------------------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``refPhaseDelay``
     - Phase delay (from ``estimate_phase_delay``)
   * - ``refPhaseDelayFine``
     - Fine phase delay
   * - ``att_uc``
     - Upconverter (DAC) attenuator, dB
   * - ``att_dc``
     - Downconverter (ADC) attenuator, dB
   * - ``amplitude_scale``
     - Probe tone power (DAC scale)

Bad Mask
--------

Exclude frequency ranges (MHz) from tuning:

.. code-block:: json

   "bad_mask": {"0": [5000, 5100], "1": [5171.64, 5171.74]}

Constants
---------

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``pA_per_phi0``
     - Phi0-to-pA conversion (chip-dependent)
   * - ``R_sh``
     - Shunt resistance (Ohms)
   * - ``bias_line_resistance``
     - Wiring resistance, cryocard to TES (Ohms)
   * - ``high_low_current_ratio``
     - High-current / low-current gain ratio

Paths
-----

- ``tune_dir`` -- tuning file storage
- ``default_data_dir`` -- output data directory

Other Sections
--------------

- **amplifier** -- 4K/50K bias values + voltage-to-DAC conversion
- **bias_group_to_pair** -- maps bias groups to bipolar DAC pairs
- **timing** -- ``"ext_ref"`` or ``"backplane"``

Firmware Defaults
-----------------

The rogue server loads a ``defaults.yml`` (from ``smurf_cfg/defaults/``)
that sets firmware-level parameters (clocks, JESD, LO frequency).
Selected automatically based on detected hardware. Naming:
``defaults_<carrier_rev>_<bay0_type>_<bay1_type>.yml``
