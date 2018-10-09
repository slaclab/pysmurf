#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pysmurf

cfg_filename = 'experiment_fp28.cfg'


"""
A function that mimics mce_cmd. This allows the user to run specific pysmurf
commands from the command line.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Stupid log command
    parser.add_argument('--log', help='Logs input phrase', default=None,
                       action='store')
    
    # TES bias commands
    parser.add_argument('--tes-bias', action='store_true', default=False,
        help='Set the tes bias. Must also set --bias-group and --bias-voltage')
    parser.add_argument('--bias-group', action='store', default=-1, type=int,
        help='The bias group to set the TES bias', choices=np.arange(8, dtype=int))
    parser.add_argument('--bias-voltage', action='store', default=-1,type=float,
        help='The bias voltage to set')


    # IV commands
    parser.add_argument('--slow-iv', action='store_true', default=False,
        help='Take IV curve using the slow method.')
    parser.add_argument('--iv-band', action='store', type=int, default=-1,
        help='The band to take the IV curve in')
    parser.add_argument('--iv-wait-time', action='store', type=float,
        default=.1, help='The wait time between steps in bias for IV taking')
    parser.add_argument('--iv-bias-high', action='store', type=float,
        default=19.9, help='The high bias in volts.')
    parser.add_argument('--iv-bias-low', action='store', type=float,
        default=0., help='The low bias in volts.')
    parser.add_argument('--iv-high-current-wait', action='store', type=float,
        default=.25, help='The time in seconds to wait in the high current mode')
    parser.add_argument('--iv-bias-step', action='store', type=float, default=.1,
        help='The bias step amplitude in units of volts.')

    # Tuning
    parser.add_argument('--tune', action='store_true', default=False,
        help='Run tuning')
    parser.add_argument('--tune-band', action='store', type=int, default=-1,
        help='The band to tune.')
    parser.add_argument('--tune-make-plot', action='store_true', default=False,
        help='Whether to make plots for tuning. This is slow.')

    # Extract inputs
    args = parser.parse_args()

    S = pysmurf.SmurfControl(cfg_file=os.path.join(os.path.dirname(__file__), 
        '..', 'cfg_files' , cfg_name), smurf_cmd_mode=True, setup=False)


    if args.log is not None:
        S.log(args.log)

    if args.tes_bias:
        S.set_tes_bias_bipolar(args.bias_group, args.bias_voltage, 
            write_log=True)

    if args.slow_iv:
        if args.iv_band < 0:
            S.log('Must input a valid band number using --iv-band')
        else:
            S.slow_iv(args.iv_band, wait_time=args.iv_wait_time, 
                      bias_high=args.iv_bias_high, bias_low=args.iv_bias_low,
                      high_current_wait=args.iv_high_current_wait, 
                      bias_step=args.iv_bias_step)

    if args.tune:
        S.tune_band(args.tune_band, make_plot=args.tune_make_plot)

    
