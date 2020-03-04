#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
sys.path.append("../../../python")
import pysmurf.client
import time

if __name__ == "__main__":
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
    parser.add_argument("--setup", default=False,
                        action="store_true",
                        help="Whether to run setup.")
    parser.add_argument("--band", type=int, required=True,
                        help="The band to run the analysis on.")
    parser.add_argument("--reset-rate-khz", type=int, required=False,
                        default=4,
                        help="The flux ramp reset rate")
    parser.add_argument("--n-phi0", type=float, required=False,
                        default=4,
                        help="The number of phi0 per flux ramp desired.")
    parser.add_argument("--no-find-freq", default=False,
                        action="store_true",
                        help="Skip the find_freq step")
    parser.add_argument("--no-setup-notches", default=False,
                        action="store_true",
                        help="Skip the setup_notches")
    parser.add_argument("--subband-low", type=int, required=False,
                        help="The starting subband for find_freq")
    parser.add_argument("--subband-high", type=int, required=False,
                        help="The end subband for find_freq")


    args = parser.parse_args()

    #######################
    # Actual functions
    #######################
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


    def execute(status_dict, func, label, save_dict=True):
        """
        Must pass func as a lambda.
        """
        status_dict[label] = {}
        status_dict[label]['start'] = S.get_timestamp(as_int=True)
        status_dict[label]['output'] = func()
        status_dict[label]['end'] = S.get_timestamp(as_int=True)
        np.save(os.path.join(S.output_dir, "status"),
                status_dict)

        return status_dict

    # why
    status = execute(status, lambda: S.why(), 'why')

    # Setup
    if args.setup:
        status = execute(status, lambda: S.setup(), 'setup')

    # amplifier biases
    status = execute(status,
                     lambda: S.set_amplifier_bias(write_log=True),
                     'amplifier_bias')
    status = execute(status,
                     lambda: S.set_cryo_card_ps_en(write_log=True),
                     'amplifier_enable')
    
    # full band response
    status = execute(status,
                     lambda: S.full_band_resp(2, make_plot=True,
                                              save_plot=True,
                                              show_plot=False,
                                              return_plot_path=True),
                     'full_band_resp')

    # find_freq
    if not args.no_find_freq:
        subband = np.arange(10, 120)
        if args.subband_low is not None and args.subband_high is not None:
            subband = np.arange(args.subband_low, args.subband_high)
        status['subband'] = subband
        status = execute(status,
                         lambda: S.find_freq(band, subband,
                                             make_plot=True, save_plot=True),
                         'find_freq')

    # setup notches
    if not args.no_setup_notches:
        status = execute(status,
                         lambda: S.setup_notches(band,
                                                 new_master_assignment=True),
                         'setup_notches')

        status = execute(status,
                         lambda: S.plot_tune_summary(band, eta_scan=True,
                                                     show_plot=False,
                                                     save_plot=True),
                         'plot_tune_summary')

    # Actually take a tuning serial gradient descent using tune_band_serial
    status = execute(status,
                     lambda: S.run_serial_gradient_descent(band),
                     'serial_gradient_descent')

    status = execute(status,
                     lambda: S.run_serial_eta_scan(band),
                     'serial_eta_scan')

    # track
    channel = S.which_on(band)
    status = execute(status,
                     lambda: S.tracking_setup(band, channel=channel,
                                              reset_rate_khz=args.reset_rate_khz,
                                              fraction_full_scale=.5,
                                              make_plot=True, show_plot=False,
                                              nsamp=2**18, lms_gain=8,
                                              lms_freq_hz=None,
                                              meas_lms_freq=False,
                                              meas_flux_ramp_amp=True,
                                              n_phi0=args.n_phi0,
                                              feedback_start_frac=.2,
                                              feedback_end_frac=.98),
                     'tracking_setup')
                    
    # now track and check
    status = execute(status, lambda: S.check_lock(band), 'check_lock')

    # Identify bias groups
    status = execute(status,
                     lambda: S.identify_bias_groups(bias_groups=np.arange(8),
                                                    make_plot=True, show_plot=False,
                                                    save_plot=True,
                                                    update_channel_assignment=True),
                     'identify_bias_groups')

    # take data

    # read back data

    # now take data using take_noise_psd and plot stuff

    # Command and IV.


    
