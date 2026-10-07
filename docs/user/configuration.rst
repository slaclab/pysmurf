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
the file the top file inherits (and whatever that inherits, downwards), then
the top file. Mappings merge key by key; a scalar or a list replaces the whole
value below it. ``inherit`` names one path, relative to the file that says it:
a chain, not a set of parents, so no layer can be applied twice. A legacy
``.cfg`` is converted in memory and resolved as the top layer under its own
path, so provenance and any refusal name the ``.cfg`` file, and an ``inherit``
in it (the old format had none) would resolve beside it.

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
It is applied key by key: a band's ``delay`` block replaces the default's
whole, not field by field, since a delay triple is one fact about one band.
A key nobody sets and the default leaves ``null`` is refused, naming it. A key
may not contain a ``.``: the dot is how a value is addressed (``bands.4.att_uc``
below), so ``a.b: 1`` is refused -- nest it as ``a: {b: 1}``.

Usage
-----

.. code-block:: python

   S = pysmurf.client.SmurfControl(cfg_file='/path/to/slot4.yaml')
   S.setup()

   S.config.values['wiring']['R_sh']
   S.config.provenance['bands.4.att_uc']     # ('/path/to/slot4.yaml', 3)
   S.config.get('bands.4.att_uc')            # the same value by dotted path
   S.config.hash                             # the same for any layering with the same values

Every value has provenance. One a band took from ``band_default`` is credited
to the line that set the default; one the schema filled in, such as a band's
firmware ``data_out_mux``, is ``('<validated>', 0)``.

Reattaching without a file
--------------------------

``setup()`` records the resolved configuration on the server, and a client
started later with no file reads it back from there:

.. code-block:: python

   S = pysmurf.client.SmurfControl()      # online, no cfg_file

The server carries it while it stays up. A server that has restarted has
forgotten, and is not configured; a configured server may hold no record
either -- a ``setup()`` that failed after clearing it, or an image without the
node it is kept in. In both cases the client raises
``RuntimeError`` saying which, asking for ``setup()`` with a file. Nothing is
guessed from disk, and nothing writes
back into the files you gave. The record is written when ``setup()``
succeeds and cleared when it starts, so a ``setup(force_configure=True)``
that fails part-way leaves nothing to reattach to -- the hardware is no
longer in the recorded configuration, and a new client is refused the same
way until a ``setup()`` completes.

What is recorded is the configuration the client was given, as the file
resolved: the same thing a client constructed from that file holds before it
calls ``setup()``. A value changed on the instance afterwards -- ``S.feedback_gain[4]
= 512`` before ``setup()``, or a tuning that writes ``S.lms_freq_hz[band]`` --
is applied to the hardware but is not configuration and is not in the record.
Values an operation measures will be kept separately.

``setup()`` also writes the resolution to a record under ``paths.status``
(``resolved/<host>_<port>.json``, one file per endpoint, with a dated copy),
together with the firmware identity and the witness registers as they read
at that moment. It is a record of what the system was given and when -- the
answer to "what was slot 4 running yesterday" -- and nothing reads it back
into a client.

Reference
---------

Every key the schema knows, by section, with its meaning and units and the
name it had in a legacy ``.cfg`` file where that differs. A key marked
*required* has no default: the site's file must set it. Where a value is read
by one method in particular, that method is named; the per-band keys are the
values ``setup()`` writes to the firmware for each band it configures.

paths
^^^^^

``data``
    Root of the data tree, ``/data/smurf_data`` by default. Each client
    session makes a dated directory under it for its outputs and plots.
    Legacy top-level ``default_data_dir``.
``smurf_cmd``
    Where ``smurf_cmd.py`` (the command-line interface) writes instead of a
    dated session directory. Legacy top-level ``smurf_cmd_dir``.
``tune``
    Where tune files are written and looked for. Legacy top-level
    ``tune_dir``.
``status``
    Where status dumps and the configuration records go. Legacy top-level
    ``status_dir``.

wiring
^^^^^^

The TES bias chain and the readout wiring, assumed the same for every
channel. Legacy: top-level keys of the same name, except where noted.

``R_sh`` (ohms, *required*)
    Resistance of the TES shunt resistors. Read by the IV analysis
    (``analyze_iv``, ``run_iv``, ``partial_load_curve_all``), the noise
    analysis and ``bias_bump``.
``bias_line_resistance`` (ohms, *required*)
    Total low-current-mode TES bias line resistance: the inline resistance on
    the cryostat card with its relays in the low-current position, plus the
    cryocable (including any cold resistors). The TES and shunt themselves
    are usually negligible beside it. Read by ``analyze_iv``, ``bias_bump``,
    ``identify_bias_groups``.
``high_low_current_ratio`` (unitless, *required*)
    Ratio of the current sourced by the cryostat card for the same applied
    bias voltage in high- versus low-current relay mode; greater than one,
    and well approximated by the ratio of the card's low- to high-current
    path resistances. Read wherever a bias is converted to a current.
``high_current_mode`` (0 or 1)
    Whether ``smurf_cmd.py`` biases in high-current mode. Legacy
    ``high_current_mode_bool``.
``pA_per_phi0`` (pA per Φ₀)
    Conversion from demodulated SQUID phase to TES current, the same for every
    channel; 9×10⁶ by default. Read by the IV and noise analyses and
    ``bias_bump``. Legacy ``constant:pA_per_phi0``.
``pic_to_bias_group`` (``{pic_channel: bias_group}``, *required*)
    Which TES bias group each cryostat-card PIC channel drives, the same
    mapping as the legacy file's; exposed to analysis code as the ``(n, 2)``
    array ``S.pic_to_bias_group``.
``bias_group_to_pair`` (``{bias_group: [dac_plus, dac_minus]}``, *required*)
    The bipolar RTM DAC pair behind each TES bias group, the same mapping as
    the legacy file's; exposed as the ``(n, 3)`` array ``S.bias_group_to_pair``
    with the group first. ``S.n_bias_groups`` is
    the number of groups here.
``all_bias_groups`` (list of int, *required*)
    The bias groups this system drives, each in ``[0, 16)``; what
    ``run_iv`` and ``overbias_tes_all`` iterate over, as ``S.all_groups``.
``bad_mask`` (``[[lo_MHz, hi_MHz], ...]``)
    RF frequency intervals in which resonator candidates are ignored by
    ``relock`` -- which ``setup_notches`` and ``track_and_check`` call --
    and so by most tuning. Exposed as the ``(n, 2)`` array ``S.bad_mask``.
    Legacy: a mapping whose labels nothing read; the intervals are the value.

attenuator
^^^^^^^^^^

``att1`` … ``att4`` (band number, *required*)
    Which 500 MHz band each of the four RF attenuators on a carrier bay
    serves. Only bands 0–3 are named: the mapping is the same on both bays
    (band modulo 4). Read by ``band_to_att`` and ``att_to_band``, which
    every ``att_uc``/``att_dc`` write goes through.

amplifier
^^^^^^^^^

The cryostat-card amplifier biasing, read as a mapping by the amplifier
commands (``set_amplifier_bias``, ``set_hemt_gate_voltage``,
``get_hemt_drain_current`` and their 50 K counterparts). The keys are the
legacy ones unchanged.

``hemt_Vg``, ``LNA_Vg`` (volts, *required*)
    Desired 4 K HEMT and 50 K LNA gate voltages at the output of the
    cryostat card; what ``set_amplifier_bias`` applies.
``bit_to_V_hemt``, ``bit_to_V_50k`` (volts per bit, *required*)
    Conversion from the RTM DAC's digital value to the gate voltage at the
    cryostat-card output. Depends on the card's voltage divider, so it is
    the card's, not the DAC's.
``dac_num_50k`` (1–32, *required*)
    The RTM DAC wired to the 50 K LNA gate, numbered as on the RTM
    schematic (``DAC1`` … ``DAC32``); DAC32 on cryostat card C02 with JMP4
    populated.
``hemt_Vd_series_resistor``, ``50K_amp_Vd_series_resistor`` (ohms)
    The resistor inline with each amplifier's drain supply, before the
    regulator, from which the drain current is inferred. 200 Ω (R44) and
    10 Ω (R54) on cryostat card revision C02.
``hemt_Id_offset``, ``50k_Id_offset`` (mA, *required*)
    The current the DC/DC regulator itself draws through that resistor,
    subtracted from the measured total to give the amplifier's drain current.
``hemt_gate_min_voltage``, ``hemt_gate_max_voltage`` (volts, *required*)
    Software limits on the 4 K HEMT gate voltage; ``set_hemt_gate_voltage``
    refuses a value outside them unless told to ``override``.
``hemt``, ``50k``
    Per-amplifier addressing for the two-amplifier cryostat card: the drain
    op-amp gain, the PIC address of the drain monitor, the power-enable
    bitmask, and optionally the gate DAC number. Carried as the legacy mapping,
    unchecked beyond ``gate_dac_num``, as the old loader carried it.
``hemt1``, ``hemt2``, ``50k1``, ``50k2``
    The same for the four-amplifier card, each with the drain DAC number and
    its volts-to-DAC conversion (``drain_conversion_m``/``_b``), the drain
    sense resistor, default and limit drain and gate voltages, and the gate
    DAC's bit-to-volt conversion.

flux_ramp, timing, fs, dsp_enable, ultrascale_temperature_limit_degC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``flux_ramp.num_flux_ramp_counter_bits`` (20 or 32, *required*)
    Width of the firmware's flux ramp counter, which sets how a ramp rate is
    turned into a DAC step size in ``flux_ramp_setup``.
``timing.timing_reference`` (``ext_ref``, ``backplane`` or ``fiber``, *required*)
    The timing source ``setup()`` selects with ``set_timing_mode``; the
    ``timing`` section is the legacy one unchanged.
``fs`` (Hz, *required*)
    The sample rate the noise analysis assumes when it is not told one.
``dsp_enable`` (0 or 1)
    Whether ``setup()`` enables baseband DSP -- tone generation, tracking,
    feedback and streaming -- on each band it configures.
``ultrascale_temperature_limit_degC`` (°C, or ``null``)
    If set, ``setup()`` arms the FPGA over-temperature shutdown at this
    limit. Legacy top-level ``ultrascale_temperature_limit_degC``.

tune
^^^^

``default_tune`` (path, or ``null``)
    A tune file to load when the client starts. Legacy
    ``tune_band:default_tune``.
``fraction_full_scale`` (0–1, *required*)
    The flux ramp amplitude as a fraction of the DAC's full scale, used by
    ``tracking_setup`` and ``flux_ramp_setup`` when not given one.
    ``S.fraction_full_scale`` is writable: ``tracking_setup`` stores the
    value it settled on. Legacy ``tune_band:fraction_full_scale``.
``reset_rate_khz`` (kHz, *required*)
    The flux ramp reset rate ``setup()`` and ``tracking_setup`` use when not
    given one. Legacy ``tune_band:reset_rate_khz``.

band_default and bands
^^^^^^^^^^^^^^^^^^^^^^

One block per band under ``bands``, keyed by band number; ``band_default``
is applied under each block first. Every key below is exposed as a
``{band: value}`` dictionary (``S.att_uc[4]``) that tuning code reads and,
for some, writes into. Legacy: ``init:band_#`` for the firmware settings and
``tune_band:<key>:<band>`` for the tuning parameters.

``iq_swap_in``, ``iq_swap_out`` (0 or 1)
    Swap I and Q at the analysis filter bank input / synthesis filter bank
    output, flipping the spectrum about the band centre; corrects a
    hardware-dependent sideband convention.
``feedback_enable`` (0 or 1), ``feedback_polarity`` (0 or 1)
    Whether the tone-tracking loop applies frequency corrections to the
    band's channels, and the sign of the correction (which depends on the eta
    calibration's convention and the wiring).
``feedback_gain`` (0–65535, *required*)
    Integral gain of the tracking loop: scales the frequency error before it
    accumulates into each tone's frequency correction. Distinct from
    ``lms_gain``.
``feedback_limit_khz`` (kHz, *required*)
    Maximum excursion of a tone from its programmed centre frequency; the
    accumulated feedback is clamped at it.
``att_uc``, ``att_dc`` (attenuator steps, *required*)
    The up-converter (tone output) and down-converter (return path)
    attenuator settings for the band, written through the ``attenuator``
    mapping above.
``amplitude_scale`` (tone power, 0–15, *required*)
    The tone amplitude tuning starts from; ``setup_notches`` and the like
    write the per-channel power they chose back into this dictionary.
``data_out_mux`` (``[lane, lane]``)
    Which two JESD transmit lanes carry the band's DAC data. Firmware-fixed
    per band and filled in by the schema (bands 0/4 → ``[2, 3]``, 1/5 →
    ``[0, 1]``, 2/6 → ``[6, 7]``, 3/7 → ``[8, 9]``); set it only to override.
``trigger_reset_delay`` (processing-clock ticks, *required*)
    Delay between the flux ramp reset trigger and the integrator reset,
    adjusted so the reset lands on the ramp's glitch.
``lms_gain`` (0–7, *required*)
    Adaptation rate of the LMS estimator of the flux ramp harmonics, applied
    as a power-of-two shift (effective gain 2^value).
``lms_freq_hz`` (Hz)
    The tracking demodulation frequency: flux ramp rate × flux quanta per
    ramp. ``tracking_setup`` measures it when asked and writes the result
    into this dictionary. Legacy ``lms_freq``.
``delta_freq`` (MHz)
    Half-width of the window around a resonance that ``eta_estimator``
    and ``find_peak`` fit in.
``feedback_start_frac``, ``feedback_end_frac`` (fraction of the ramp)
    The part of each flux ramp cycle, each in ``[0, 1]``, within which the
    tracking feedback is applied; ``tracking_setup`` judges the pair when it
    runs.
``gradient_descent_gain``, ``_averages``, ``_converge_hz``, ``_step_hz``, ``_momentum``, ``_beta``
    The serial gradient descent's learning rate, measurements averaged per
    gradient sample, convergence threshold (Hz), probe offset (Hz),
    optimiser mode (1 momentum, 0 adaptive step) and running-average decay.
``eta_scan_averages``, ``eta_scan_del_f`` (count; Hz)
    For ``run_serial_eta_scan``: frequency-error samples averaged per point,
    and the offset about each resonator at which the error is sampled.
``band_delay_us`` (microseconds) or ``delay``
    The round-trip delay compensation, one of two ways; see below.

The band delay is given one of two ways. ``band_delay_us`` is the total in
microseconds, from ``S.estimate_phase_delay(band)``, and the firmware derives
its registers from it. A ``delay`` block sets those registers directly and
wins when present::

   bands:
     4:
       delay: {ref_phase: 6, ref_phase_fine: 0, lms: 24}

``ref_phase`` is the coarse round-trip delay (``refPhaseDelay``) in
processing-clock ticks -- 2.4 MHz ticks on current firmware, so 6 is 2.5 µs;
``ref_phase_fine`` (``refPhaseDelayFine``) adds a lag to the DAC output in
307.2 MHz ticks and so *subtracts* from the total; ``lms`` (``lmsDelay``)
aligns the feedback with the flux ramp phase and is typically equal to
``ref_phase``. ``lms`` left out means the same value as ``ref_phase``;
``ref_phase`` itself is at least 1 -- a zero triple was never written, a
band with no delay gives ``band_delay_us``. Converted legacy files carry their ``refPhaseDelay``/``refPhaseDelayFine``/
``lmsDelay`` as a ``delay`` block, so what ``setup()`` writes does not
change. Both include the digital delay, so they vary with firmware version.

Firmware Defaults
-----------------

The rogue server loads a ``defaults.yml`` (from ``smurf_cfg/defaults/``)
that sets firmware-level parameters (clocks, JESD, LO frequency).
Selected automatically based on detected hardware. Naming:
``defaults_<carrier_rev>_<bay0_type>_<bay1_type>.yml``. That is the
platform's configuration and is not part of this file.
