Operations
==========

Detailed reference for common operations. See :doc:`concepts` for
the underlying algorithms.

Tuning
------

find_freq
~~~~~~~~~

Sweeps sub-bands to find resonance dips:

.. code-block:: python

   S.find_freq(band, start_freq=-250, stop_freq=250)

setup_notches
~~~~~~~~~~~~~

Places tones at found resonances and does coarse eta estimation:

.. code-block:: python

   S.setup_notches(band, new_master_assignment=True)

eta_scan
~~~~~~~~

Measures the eta parameter per channel (written to firmware as
etaMag/etaPhase in BRAM bank 0):

.. code-block:: python

   eta = S.eta_scan(band, subband, freq, tone_power)

track_and_check
~~~~~~~~~~~~~~~

Enables feedback (sets ``feedbackEnable``), verifies channels lock,
prunes failures:

.. code-block:: python

   S.track_and_check(band)

Feedback Control
----------------

.. code-block:: python

   S.set_feedback_enable(band, 1)       # enable tracking
   S.set_feedback_gain(band, val)       # lmsGain (loop bandwidth)
   S.set_feedback_limit(band, val)      # saturation limit (Hz)
   S.set_feedback_polarity(band, val)   # loop sign

Firmware gain = ``lmsGain`` (global) x per-channel ``feedbackGain``.

Flux Ramp
---------

.. code-block:: python

   S.flux_ramp_on()
   S.set_flux_ramp_freq(4000)       # Hz (reset rate)
   S.set_flux_ramp_dac(0x4000)      # amplitude
   S.flux_ramp_off()

The flux ramp rate sets the detector sample rate. The phi0 rate
(reset_rate x phi0/ramp) determines the tracking carrier frequency.

Streaming
---------

.. code-block:: python

   # Timed acquisition
   datafile = S.take_stream_data(meas_time=30, downsample_factor=200)

   # Manual control
   S.stream_data_on()
   S.stream_data_off()

   # Context manager
   with S.stream_data_cm():
       pass

   # Read back
   timestamps, phase, mask = S.read_stream_data(datafile)

The raw firmware frame rate is ~4 kHz. Default software filter:
``butter(4, 2*63/4000)``. With ``downsample_factor=200``, output is
~20 Hz.

Debug Data
~~~~~~~~~~

Single-channel diagnostics from firmware streaming BRAM:

.. code-block:: python

   f, df, sync = S.take_debug_data(band, channel=42)

ADC/DAC
~~~~~~~

.. code-block:: python

   S.read_adc_data(band)
   S.read_dac_data(band)
   S.check_adc_saturation(band)
   S.check_dac_saturation(band)

TES Bias and IV Curves
----------------------

.. code-block:: python

   S.set_tes_bias(bias_group, voltage)
   S.get_tes_bias(bias_group)

   S.run_iv(
       bias_groups=[0, 1, 2, 3],
       bias_high=19.9, bias_low=0, bias_step=0.025,
       overbias_voltage=19.9, wait_time=0.1,
   )

IV curves overbias the TES then step down through the transition to
find optimal bias points.

Amplifier Bias
--------------

.. code-block:: python

   S.set_amplifier_bias(bias_hemt=0.6, bias_50k=0.02)
   S.get_amplifier_biases()

Noise
-----

.. code-block:: python

   S.take_noise_psd(meas_time=60, band=band, show_plot=True)

Computes PSD and fits white + 1/f. Typical well-tuned white noise:
50--100 pA/rtHz at TES.

Saving State
------------

.. code-block:: python

   S.save_tune()
   S.tune(band, load_tune=True)  # next session
