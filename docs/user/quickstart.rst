Quickstart
==========

Prerequisites: a running rogue server, a config file (see
:doc:`configuration`), and pysmurf installed.

Connect
-------

.. code-block:: python

   import pysmurf
   S = pysmurf.SmurfControl(cfg_file='/path/to/config.cfg', make_logfile=True)

Setup
-----

.. code-block:: python

   band = 0
   S.set_att_uc(band, 18)   # dB, upconverter
   S.set_att_dc(band, 18)   # dB, downconverter
   S.estimate_phase_delay(band)  # once per hardware setup

Tune
----

.. code-block:: python

   S.tune(band, load_tune=False)
   # Runs: find_freq -> setup_notches -> serial eta scan -> track_and_check

   # Or reload a previous tune:
   S.tune(band, load_tune=True, tune_file='/path/to/tune.npy')

Check Channels
--------------

.. code-block:: python

   channels = S.which_on(band)
   print(f"{len(channels)} channels tracking")

Stream Data
-----------

.. code-block:: python

   datafile = S.take_stream_data(meas_time=30)
   timestamps, phase, mask = S.read_stream_data(datafile)

Noise
-----

.. code-block:: python

   S.take_noise_psd(meas_time=60, band=band, show_plot=True, save_plot=True)

IV Curves
---------

.. code-block:: python

   S.run_iv(
       bias_groups=[0, 1, 2, 3],
       bias_high=19.9, bias_low=0, bias_step=0.025,
       overbias_voltage=19.9, wait_time=0.1,
   )

Shutdown
--------

.. code-block:: python

   S.band_off(band)
