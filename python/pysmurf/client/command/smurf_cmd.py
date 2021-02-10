#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File       : pysmurf/command/smurf_cmd.py
# Created    : 2018-10-05
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import argparse
import os
import sys
import time

import numpy as np

import pysmurf.client

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


cfg_filename = os.path.join('/usr/local/src/pysmurf/', 'cfg_files', 'stanford',
                            'experiment_fp31_cc03-02_lbOnlyBay0.cfg')

"""
A function that mimics mce_cmd. This allows the user to run specific pysmurf
commands from the command line.
"""
def make_runfile(output_dir, row_len=60, num_rows=60, data_rate=60,
    num_rows_reported=60):
    """
    Make the runfile
    """
    S = pysmurf.client.SmurfControl(cfg_file=cfg_filename, smurf_cmd_mode=True, setup=False)

    S.log('Making Runfile')

    with open(os.path.join(os.path.dirname(__file__),
            "runfile/runfile.default.bicep53")) as f:
        lines = f.readlines()
        line_holder = []
        for line in lines:
            # A bunch of replacements
            if "ctime=<replace>" in line:
                timestamp = S.get_timestamp()
                S.log(f'Adding ctime {timestamp}')
                line = line.replace("ctime=<replace>", f'ctime={timestamp}')
            elif "Date:<replace>" in line:
                time_string = time.strftime("%a %b %d %H:%M:%S %Y",
                    time.localtime())
                S.log(f'Adding date {time_string}')
                line = line.replace('Date:<replace>', f'Date: {time_string}')
            elif "row_len : <replace>" in line:
                S.log(f"Adding row_len {row_len}")
                line = line.replace('row_len : <replace>',
                                    f'row_len : {row_len}')
            elif "num_rows : <replace>" in line:
                S.log(f"Adding num_rows {num_rows}")
                line = line.replace('num_rows : <replace>',
                                    f'num_rows : {num_rows}')
            elif "num_rows_reported : <replace>" in line:
                S.log(f"Adding num_rows_reported {num_rows_reported}")
                line = line.replace('num_rows_reported : <replace>',
                                    f'num_rows_reported : {num_rows_reported}')
            elif "data_rate : <replace>" in line:
                S.log(f"Adding data_rate {data_rate}")
                line = line.replace('data_rate : <replace>',
                                    f'data_rate : {data_rate}')
            line_holder.append(line)

    full_path = os.path.join(output_dir,
        f'smurf_status_{S.get_timestamp()}.txt')

    for line in line_holder:
        print(line)
    with open(full_path, "w") as f1:
        f1.writelines(line_holder)

    S.log(f"Writing to {full_path}")
    sys.stdout.writelines(line_holder)

def start_acq(S):
    """
    Start an acquisition

    Args
    ----
    S : SmurfControl
        The SmurfControl object used to issue commands
    """
    # Need to build out streaming header
    S.log('Starting streaming data')
    S.set_stream_enable(1, write_log=True)

def stop_acq(S):
    """
    Stop the acquisition

    Args
    ----
    S : SmurfControl
        The SmurfControl object used to issue commands
    """
    np.array(S.config.get('init').get('bands'))
    S.log('Stopping streaming data')
    S.set_stream_enable(False)

def acq_n_frames(S, n_frames):
    """
    Sends the amount of data requested by the user in units of n_frames.

    Args
    ----
    S : SmurfControl
        The SmurfControl object used to issue commands
    n_frames : int
        The number of frames to keep data streaming on.
    """
    start_acq(S)
    make_runfile(S.output_dir, num_rows=num_rows, data_rate=data_rate,
        row_len=row_len, num_rows_reported=num_rows_reported)
    sample_rate = 50E6 / num_rows / data_rate / row_len
    wait_time = n_frames / sample_rate
    time.sleep(wait_time)
    stop_acq(S)


def set_port(S, slot, port):
    """
    Define a port/slot pair.

    Args
    ----
    S : SmurfControl
        The SmurfControl object used to issue commands
    slot : int
        The number of SMuRF slot
    """
    slot_port_file = os.path.join(S.output_dir, 'slot_port_def.txt')
    slots, ports = np.loadtxt(slot_port_file, dtype=int).T


    if slot in slots:
        # Change port number if it already exists
        idx = np.where(slots == slot)[0][0]
        ports[idx] = port
    else:
        # Append the slot/port pairs
        slots = np.append(slots, slot)
        ports = np.append(ports, port)

    np.savetxt(slot_port_file, np.array([slots, ports]), fmt='%i %i')


def get_port(S, slot):
    """
    Get the port number for streaming

    Args
    ----
    S : SmurfControl
        The SmurfControl object used to issue commands
    slot : int
        The number of SMuRF slot

    Returns
    -------
    port : int
        The port number associated with slot to stream data.
    """
    slot_port_file = os.path.join(S.output_dir, 'slot_port_def.txt')

    # Load the data
    slots, ports = np.loadtxt(slot_port_file, dtype=int).T

    idx = np.where(slots == slot)[0]
    if len(idx) == 0:
        raise ValueError("Slot is not in the port/slot defintion file. " +
            "Update file with specify_port function.")

    return ports[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epics-prefix', help='The epics root',
                        action='store', default=None, type=str)
    # Offline mode
    parser.add_argument('--offline', help='For offline debugging',
        default=False, action='store_true')

    # Stupid log command
    parser.add_argument('--log', help='Logs input phrase', default=None,
        action='store')

    # TES bias commands
    parser.add_argument('--tes-bias', action='store_true', default=False,
        help='Set the tes bias. Must also set --bias-group and --bias-voltage')
    parser.add_argument('--bias-group', action='store', default=-1, type=int,
        help='The bias group to set the TES bias. If -1, then sets all.',
        choices=np.array([-1,0,1,2,3,4,5,6,7], dtype=int))
    parser.add_argument('--bias-voltage', action='store', default=0.,
        type=float, help='The bias voltage to set')
    parser.add_argument('--bias-voltage-array', action='store',
        default=None, help='Array of voltages to set per bias group')

    parser.add_argument('--overbias-tes', action='store_true', default=False,
        help='Overbias the TESs')
    parser.add_argument('--overbias-tes-wait', action='store', default=1.5,
        type=float, help='The time to stay at the high current.')

    parser.add_argument('--bias-bump', action='store', default=False,
        help='Bump the TES bias up and down')
    parser.add_argument('--bias-bump-step', action='store', default=0.01,
        type=float, help="Size of bias bump step in volts")
    parser.add_argument('--bias-bump-wait', action='store', default=5.,
        type=float, help="Time to dwell at stepped up/down bias")
    parser.add_argument('--bias-bump-between', action='store', default=3.,
        type=float, help="Interval between up and down steps.")

    # IV commands
    parser.add_argument('--run-iv', action='store_true', default=False,
        help='Take IV curve.')
    parser.add_argument('--plc', action='store_true', default=False,
        help='Take partial load curve.')
    parser.add_argument('--iv-band', action='store', type=int, default=-1,
        help='The band to take the IV curve in')
    parser.add_argument('--iv-wait-time', action='store', type=float,
        default=.1, help='The wait time between steps in bias for IV taking')
    parser.add_argument('--iv-bias-high', action='store', type=float,
        default=19.9, help='The high bias in volts.')
    parser.add_argument('--iv-bias-low', action='store', type=float,
        default=0., help='The low bias in volts.')
    parser.add_argument('--iv-high-current-wait', action='store', type=float,
        default=.5, help='The time in seconds to wait in the high current mode')
    parser.add_argument('--iv-bias-step', action='store', type=float, default=.1,
        help='The bias step amplitude in units of volts.')

    # Tuning
    parser.add_argument('--tune', action='store_true', default=False,
        help='Run tuning')
    parser.add_argument('--tune-make-plot', action='store_true', default=False,
        help='Whether to make plots for tuning. This is slow.')
    parser.add_argument('--last-tune', action='store_true', default=False,
        help='Use the last tuning')
    parser.add_argument('--use-tune', action='store', type=str, default=None,
        help='The full path of a tuning file to use.')
    parser.add_argument('--check-lock', action='store', default=False,
        help='Check tracking and kill unlocked channels.')

    # Start acq
    parser.add_argument('--start-acq', action='store_true', default=False,
        help='Start the data acquisition')
    parser.add_argument('--row-len', action='store', default=60, type=int,
        help='The variable to stuff into the runfile. See the MCE wiki')
    parser.add_argument('--num-rows', action='store', default=33, type=int,
        help='The variable to stuff into the runfile. See the MCE wiki')
    parser.add_argument('--num-rows-reported', action='store', default=33,
        type=int,
        help='The variable to stuff into the runfile. See the MCE wiki')
    parser.add_argument('--data-rate', action='store', default=60, type=int,
        help='The variable to stuff into the runfile. See the MCE wiki')
    parser.add_argument('--n-frames', action='store', default=-1, type=int,
        help='The number of frames to acquire.')

    parser.add_argument('--make-runfile', action='store_true', default=False,
        help='Make a new runfile.')

    parser.add_argument('--status', action='store_true', default=False,
        help='Dump status to screen')

    # Stop acq
    parser.add_argument('--stop-acq', action='store_true', default=False,
        help='Stop the data acquistion')

    # Turning stuff off
    parser.add_argument('--flux-ramp-off', action='store_true', default=False,
        help='Turn off flux ramp')
    parser.add_argument('--all-off', action='store_true', default=False,
        help='Turn off everything (tones, TES biases, flux ramp)')

    # Soft reset
    parser.add_argument('--soft-reset', action='store_true', default=False,
        help='Soft reset SMuRF.')

    # Setup, in case smurf went down and you have to start over
    # do we want this to be folded into tuning with an if statement?
    # separate for now
    # CY 20181125
    parser.add_argument('--setup', action='store_true', default=False,
        help='Setup SMuRF and load defaults.')

    # Defining slot/port paris
    parser.add_argument('--get-port', action='store_true', default=False,
        help='Get the port number. Must define slot')
    parser.add_argument('--set-port', action='store_true', default=False,
        help='Set the port number. Bust define slot and port')
    parser.add_argument('--port', action='store', default=-1, type=int,
        help='The port number for get/set port')
    parser.add_argument('--slot', action='store', default=-1, type=int,
        help='The slot number used for get/set port')

    # Extract inputs
    args = parser.parse_args()

    offline = args.offline

    # Check for too many commands
    n_cmds = (args.log is not None) + args.tes_bias + args.run_iv + \
        args.plc + args.tune + args.start_acq + args.stop_acq + \
        args.last_tune + (args.use_tune is not None) + args.overbias_tes + \
        args.bias_bump + args.soft_reset + args.make_runfile + args.setup + \
        args.flux_ramp_off + args.all_off + args.check_lock + args.status
    if n_cmds > 1:
        sys.exit(0)

    epics_prefix = args.epics_prefix
    S = pysmurf.client.SmurfControl(epics_root=epics_prefix,
                                    cfg_file=cfg_filename, smurf_cmd_mode=True,
                                    setup=False, offline=offline)

    if args.log is not None:
        S.log(args.log)

    ### Tuning related commands ###
    if args.last_tune:
        S.log('Loading in last tuning')
        S.tune(last_tune=True, make_plot=args.tune_make_plot)

    if args.use_tune is not None:
        S.log(f'Loading old tune from file: {args.use_tune}')
        S.tune(tune_file = args.use_tune, make_plot=args.tune_make_plot)

    if args.tune:
        S.log('Running a smurf tuning. Using old frequency file but new eta scan')
        S.tune(retune=True, make_plot=args.tune_make_plot)

    if args.check_lock:
        S.log('Running track and check')
        for band in S.config.get('init')['bands']:
            S.check_lock(band) # this might be too slow

    ### TES bias related commands ###
    if args.tes_bias:
        bias_voltage = args.bias_voltage
        if args.bias_group == -1:
            if args.bias_voltage_array is not None:
                bias_voltage_str = args.bias_voltage_array
                bias_voltage_array = [float(bias) for bias in bias_voltage_str.split(" ")]
                bias_voltage_array = np.array(bias_voltage_array)
            else:
                bias_voltage_array = np.zeros((S._n_biasgroups,))
                bias_voltage_array[S.all_groups] = bias_voltage # all_groups from cfg
            S.set_tes_bias_bipolar_array(bias_voltage_array, write_log=True)
        else:
            S.set_tes_bias_bipolar(args.bias_group, bias_voltage,
                write_log=True)

    if args.overbias_tes:

        if S.high_current_mode_bool: # best thing I could think of, sorry -CY
            tes_bias = 2. # drop down to 2V to wait
        else:
            tes_bias = 19.9 # factor of 10ish from above

        if args.bias_group < 0:
            S.overbias_tes_all(overbias_wait=args.overbias_tes_wait,
                bias_groups=S.all_groups,
                high_current_mode=S.high_current_mode_bool, tes_bias=tes_bias)
        else:
            S.overbias_tes(args.bias_group,
                overbias_wait=args.overbias_tes_wait,
                high_current_mode=S.high_current_mode_bool, tes_bias=tes_bias)

    if args.run_iv:
        iv_bias_high = args.iv_bias_high
        iv_bias_low = args.iv_bias_low
        iv_bias_step = args.iv_bias_step

        S.log(f'bias high {iv_bias_high}')
        S.log(f'bias low {iv_bias_low}')
        S.log(f'bias step {iv_bias_step}')
        # 20181223: CY took out IV biases in terms of current. Revert if you
        #  decide you want it back

        if iv_bias_high > 19.9:
            iv_bias_high = 19.9

        if iv_bias_step < 0:
            iv_bias_step = np.abs(iv_bias_step)

        if args.bias_group < 0: # all
            S.log('running IV on all bias groups')
            S.run_iv(bias_groups=S.all_groups, wait_time=args.iv_wait_time,
                bias_high=iv_bias_high, bias_low = iv_bias_low,
                high_current_wait=args.iv_high_current_wait,
                high_current_mode=S.high_current_mode_bool,
                bias_step=iv_bias_step, make_plot=False)
        else: # individual bias group
            S.log(f'running IV on bias group {args.bias_group}')
            S.run_iv(bias_groups=np.array([args.bias_group]),
                wait_time=args.iv_wait_time, bias_high=iv_bias_high,
                bias_low=iv_bias_low,
                high_current_wait=args.iv_high_current_wait,
                high_current_mode=S.high_current_mode_bool,
                bias_step=iv_bias_step, make_plot=False)

    if args.bias_bump:
        S.bias_bump(bias_group=S.all_groups, gcp_mode=True,
            gcp_wait=args.bias_bump_wait, gcp_between=args.bias_bump_between,
            step_size=args.bias_bump_step) # always do this on all bias groups?

    if args.plc:
        bias_high = np.zeros((8,))
        bias_high[S.all_groups] = args.iv_bias_high
        S.log(f'plc bias high {bias_high}')
        S.log(f'plc bias low {S.get_tes_bias_bipolar_array()}')
        S.log(f'plc bias step {args.iv_bias_step}')

        iv_bias_step = np.abs(args.iv_bias_step) * 1.5 # speed this up relative to other mce's

        S.log('running plc on all bias groups')
        S.partial_load_curve_all(bias_high, bias_step=iv_bias_step,
            wait_time=args.iv_wait_time, analyze=False, make_plot=False)


    ### Turning stuff off ###
    if args.flux_ramp_off:
        S.log('Turning off flux ramp')
        S.flux_ramp_off()

    if args.all_off:
        S.log('Turning off everything')
        S.all_off()

    ### Dump smurf status
    if args.status:
        print(S.dump_state(return_screen=True))

    ### Acquistion and resetting commands ###
    if args.start_acq:

        if args.n_frames >= 1000000000:
            args.n_frames = -1 # if it is a big number then just make it continuous

        if args.n_frames > 0: # was this a typo? used to be <
            acq_n_frames(S, args.num_rows, args.num_rows_reported,
                args.data_rate, args.row_len, args.n_frames)
        else:
            S.log('Starting continuous acquisition')
            start_acq(S)
            # Don't make runfiles for now. Need to figure out
            # make_runfile(S.output_dir, num_rows=args.num_rows,
            #    data_rate=args.data_rate, row_len=args.row_len,
            #    num_rows_reported=args.num_rows_reported)
            # why are we making a runfile though? do we intend to dump it?

    if args.stop_acq:
        stop_acq(S)

    if args.soft_reset:
        S.log('Soft resetting')
        S.set_smurf_to_gcp_clear(True)
        time.sleep(.1)
        S.set_smurf_to_gcp_clear(False)

    if args.setup:
        S.log('Running setup')
        S.setup()
        # anything we need to do with setting up streaming?

    if args.make_runfile:
        make_runfile(S.output_dir, num_rows=args.num_rows,
            data_rate=args.data_rate, row_len=args.row_len,
            num_rows_reported=args.num_rows_reported)

    if args.get_port:
        if args.slot < 0 :
            raise ValueError("Must specify a slot. Use --slot")
        get_port(S, args.slot)

    if args.set_port:
        if args.slot < 0 :
            raise ValueError("Must specify a slot. Use --slot")
        if args.port < 0 :
            raise ValueError("Must specify a port. Use --port")
        set_port(S, args.slot, args.port)