.. _configuration:

Configuration
=============

pysmurf reads one YAML configuration file per physical hardware setup. The
file sits on a default shipped with the package, so it names only what differs
for this system; it may build on other files with ``inherit:``, so a site
keeps what its systems share in one place. Every resolved value can be traced
to the file and line that set it.

The legacy JSON ``.cfg`` files are still read, converted in memory with a
deprecation warning. Write one out as YAML once with::

   python -m pysmurf.client.config convert experiment.cfg -o experiment.yaml

and check what any file resolves to, and where each value came from, with::

   python -m pysmurf.client.config resolve experiment.yaml --provenance

Layers
------

Resolution applies layers lowest first: the packaged ``default.yaml``, then
each file the top file inherits (in order), then the top file. Mappings merge
key by key; a scalar or a list replaces the whole value below it. ``inherit``
names one path or a list of paths, relative to the file that says it.

.. code-block:: yaml

   # site.yaml -- what every system at this site shares
   wiring:
     R_sh: 4.0e-4
     bias_line_resistance: 16100
     high_low_current_ratio: 5.669
     pic_to_bias_group: {0: 0, 1: 1, 2: 2, 3: 3}
     bias_group_to_pair: {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8]}
     all_bias_groups: [0, 1, 2, 3]
   amplifier:
     hemt_Vg: -0.6
     LNA_Vg: -0.7
     # ...
   flux_ramp: {num_flux_ramp_counter_bits: 20}
   timing: {timing_reference: ext_ref}
   fs: 4000.0
   tune: {fraction_full_scale: 0.5, reset_rate_khz: 4.0}
   band_default:
     feedback_gain: 256
     feedback_limit_khz: 225.0
     amplitude_scale: 11
     trigger_reset_delay: 60
     lms_gain: 7
     band_delay_us: 2.4
     lms_freq_hz: 20000.0
     # ... every tuning parameter

.. code-block:: yaml

   # slot4.yaml -- one system
   inherit: site.yaml
   bands:
     4: {att_uc: 12, att_dc: 0}
     5: {att_uc: 14, att_dc: 2}

``band_default`` is applied to every band listed under ``bands`` before the
band's own block, so the two files above resolve to two fully specified bands.
A key nobody sets and the default leaves ``null`` is refused, naming it.

Usage
-----

.. code-block:: python

   S = pysmurf.SmurfControl(cfg_file='/path/to/slot4.yaml')
   S.setup()

   S.config.values['wiring']['R_sh']
   S.config.provenance['bands.4.att_uc']     # ('/path/to/slot4.yaml', 3)
   S.config.hash                             # the same for any layering with the same values

Reattaching without a file
--------------------------

``setup()`` publishes the resolved configuration to the server and to a
*sidecar* file under ``paths.status`` (``resolved/<host>_<port>.json``, with
a dated copy). A client started later with no file adopts it:

.. code-block:: python

   S = pysmurf.SmurfControl()      # online, no cfg_file

The server is asked first. If it has restarted and forgotten, the sidecar is
read and checked against the system: the firmware identity and the registers
the configuration set (ramp, trigger, streaming, per-band DSP enable). A
disagreement raises :class:`cryodaq.DescriptionMismatch` naming the register
and both values; a system nothing remembers configuring raises
``RuntimeError`` asking for ``setup()``. Deleting the sidecar and running
``setup()`` again is always a valid recovery. The sidecar is a cache of what
the server knows, not a configuration file: nothing writes back into the files
you gave.

Sections
--------

.. list-table::
   :header-rows: 1

   * - Section
     - Holds
   * - ``paths``
     - ``data``, ``smurf_cmd``, ``tune``, ``status`` directories
   * - ``wiring``
     - ``R_sh``, ``bias_line_resistance``, ``high_low_current_ratio``,
       ``high_current_mode``, ``pA_per_phi0``, ``pic_to_bias_group``,
       ``bias_group_to_pair``, ``all_bias_groups``, ``bad_mask``
       (``[[lo_MHz, hi_MHz], ...]``)
   * - ``attenuator``
     - which band each of the four RF attenuators serves
   * - ``amplifier``
     - 4K/50K bias values and voltage-to-DAC conversions, as the amplifier
       commands read them
   * - ``flux_ramp``, ``timing``, ``fs``, ``dsp_enable``
     - counter bits; ``ext_ref``, ``backplane`` or ``fiber``; sample rate
   * - ``tune``
     - ``default_tune``, ``fraction_full_scale``, ``reset_rate_khz``
   * - ``band_default``, ``bands``
     - per-band settings, below

Per-band keys
-------------

``iq_swap_in``, ``iq_swap_out``, ``feedback_enable``, ``feedback_polarity``,
``feedback_gain``, ``feedback_limit_khz``, ``att_uc``, ``att_dc``,
``amplitude_scale``, ``data_out_mux``, ``trigger_reset_delay``, ``lms_gain``,
and the tuning parameters ``lms_freq_hz``, ``delta_freq``,
``feedback_start_frac``, ``feedback_end_frac``, ``gradient_descent_*``,
``eta_scan_averages``, ``eta_scan_del_f``.

The band delay is given one of two ways. ``band_delay_us`` is the total in
microseconds, from ``S.estimate_phase_delay(band)``, and the firmware derives
its registers from it. A ``delay`` block sets those registers directly and
wins when present::

   bands:
     4:
       delay: {ref_phase: 6, ref_phase_fine: 0, lms: 24}

``lms`` left out means the same value as ``ref_phase``. Converted legacy files
carry their ``refPhaseDelay``/``refPhaseDelayFine``/``lmsDelay`` as a ``delay``
block, so what ``setup()`` writes does not change.

Firmware Defaults
-----------------

The rogue server loads a ``defaults.yml`` (from ``smurf_cfg/defaults/``)
that sets firmware-level parameters (clocks, JESD, LO frequency).
Selected automatically based on detected hardware. Naming:
``defaults_<carrier_rev>_<bay0_type>_<bay1_type>.yml``. That is the
platform's configuration and is not part of this file.
