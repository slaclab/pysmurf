#!/usr/bin/env python3
import argparse

#####################
# Arg parse things
#####################

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--epics-root", type=str, required=True,
                    help="The epics root.")
parser.add_argument("--config-file", type=str, required=True,
                    help="The configuration file to use for this test.")
parser.add_argument("--shelf-manager", type=str, required=True,
                    help="The shelf manager root.")
parser.add_argument("--setup", type=bool, required=False, default=True,
                    help="Whether to run setup.")
parser.add_argument("--band", type=int, required=True,
                    help="The band to run the analysis on.")
parser.add_argument("--reset-rate-khz", type=int, required=False,
                    default=4,
                    help="The flux ramp reset rate")
parser.add_argument("--n-phi0", type=float, required=False,
                    default=4,
                    help="The number of phi0 per flux ramp desired.")

args = parser.parse_args()

#######################
# Actual functions
#######################
import numpy as np
import sys
sys.path.append("../../../python")
import pysmurf.client

band = args.band
status = {}
status["band"] = band

# Initialize
S = pysmurf.client.SmurfControl(epics_root=args.epics_root,
                                cfg_file=args.config_file,
                                shelf_manager=args.shelf_manager,
                                setup=False)

print("All outputs going to: ")
print(S.output_dir)

# Setup
status["setup"] = args.setup
if args.setup:
    S.setup()

# Get amplifier biases
S.set_amplifier_bias(write_log=True)
S.C.write_ps_en(11)
status['amplifier'] = S.get_amplifier_bias()

# full band response
S.full_band_resp(2, make_plot=True, save_plot=True,
                 show_plot=False)

# find_freq
S.find_freq(band, make_plot=True, save_plot=True)

# setup notches
S.setup_notches(band, new_master_assignment=True)

S.plot_tune_summary(band, eta_scan=True, show_plot=False, save_plot=True)

# Actually take a tuning serial gradient descent using tune_band_serial
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

# track
channels = S.which_on(band)
S.tracking_setup(band, channel=channel, reset_rate_khz=args.reset_rate_khz,
                 fraction_full_scale=.5, make_plot=True, show_plot=False,
                 n_samp=2**18, lms_gain=8, lms_freq_hz=None,
                 meas_lms_freq=False, meas_flux_ramp_amp=True,
                 n_phi0=args.n_phi0, feedback_start_frac=.2,
                 feedback_end_frac=.98)
                    
# now track and check
S.check_lock(band)

# Identify bias groups
dd = S.identify_bias_groups(bias_groups=np.arange(8),
                            make_plot=True, show_plot=False,
                            save_plot=True,
                            update_channel_assignment=True)

# take data

# read back data

# now take data using take_noise_psd and plot stuff

# Command and IV.

# Play TES tone file
