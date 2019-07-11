#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import time
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pysmurf

cfg_filename = 'experiment_k2umux.cfg'


"""
A function that mimics mce_cmd. This allows the user to run specific pysmurf
commands from the command line.
"""
def make_runfile(output_dir, row_len=60, num_rows=60, data_rate=60,
                 num_rows_reported=60):
    """
    Make the runfile
    """
    #considering put assertions here 
    #assert isinstance ((row_len,num_rows,data_rate,num_rows_reported),int)
    #assert isinstance (output_dir,str)
    #assert row_len&num_rows&data_rate&num_rows_reported > 0
    #S.log('Making pysmurf object')
    S = pysmurf.SmurfControl(cfg_file=os.path.join(os.path.dirname(__file__), 
        '..', 'cfg_files' , cfg_filename), smurf_cmd_mode=True, setup=False)

    S.log('Making Runfile')
    
    # 20181119 dB, modified to use the correct format runfile.
    #with open(os.path.join(os.path.dirname(__file__),"runfile/runfile_template.txt")) as f:
    with open(os.path.join(os.path.dirname(__file__),
        "runfile/runfile.default.bicep53")) as f:
        lines = f.readlines()
        line_holder = []
        for l in lines:
            # A bunch of replacements
            if "ctime=<replace>" in l:
                timestamp = S.get_timestamp()
                S.log('Adding ctime {}'.format(timestamp))
                l = l.replace("ctime=<replace>", "ctime={}".format(timestamp))
            elif "Date:<replace>" in l:
                time_string = time.strftime("%a %b %d %H:%M:%S %Y", 
                    time.localtime())
                S.log('Adding date {}'.format(time_string))
                l = l.replace('Date:<replace>', "Date: {}".format(time_string))
            elif "row_len : <replace>" in l:
                S.log("Adding row_len {}".format(row_len))
                l = l.replace('row_len : <replace>', 
                    'row_len : {}'.format(row_len))
            elif "num_rows : <replace>" in l:
                S.log("Adding num_rows {}".format(num_rows))
                l = l.replace('num_rows : <replace>', 
                    'num_rows : {}'.format(num_rows))
            elif "num_rows_reported : <replace>" in l:
                S.log("Adding num_rows_reported {}".format(num_rows_reported))
                l = l.replace('num_rows_reported : <replace>', 
                    'num_rows_reported : {}'.format(num_rows_reported))
            elif "data_rate : <replace>" in l:
                S.log("Adding data_rate {}".format(data_rate))
                l = l.replace('data_rate : <replace>', 
                    'data_rate : {}'.format(data_rate))
            line_holder.append(l)
            
    full_path = os.path.join(output_dir, 
        'smurf_status_{}.txt'.format(S.get_timestamp()))

    #20181119 mod by dB to dump content of runfile, not path of runfile
    #print(full_path)
    for line in line_holder:
        print(line)
    with open(full_path, "w") as f1:
        f1.writelines(line_holder)

    S.log("Writing to {}".format(full_path))
    sys.stdout.writelines(line_holder)

def start_acq(S, num_rows, num_rows_reported, data_rate, 
    row_len):
    """
    """
    bands = S.config.get('init').get('bands')
    S.log('Setting PVs for streaming header')
    S.set_num_rows(num_rows, write_log=True)
    S.set_num_rows_reported(num_rows_reported, write_log=True)
    S.set_data_rate(data_rate, write_log=True)
    S.set_row_len(row_len, write_log=True)

    S.log('Starting streaming data')
    S.set_smurf_to_gcp_stream(True, write_log=True)
    for b in bands:
        S.set_stream_enable(b, 1, write_log=True)

def stop_acq(S):
    """
    """
    bands = np.array(S.config.get('init').get('bands'))
    S.log('Stopping streaming data')
    #for b in bands:
    #    S.set_stream_enable(b, 0)
    S.set_smurf_to_gcp_stream(False, write_log=True)

def acq_n_frames(S, num_rows, num_rows_reported, data_rate, 
    row_len, n_frames):
    """
    Sends the amount of data requested by the user in units of n_frames.

    Args:
    -----
    n_frames (int); The number of frames to keep data streaming on.
    """
    start_acq(S, num_rows, num_rows_reported, data_rate, row_len)
    make_runfile(S.output_dir, num_rows=num_rows, data_rate=data_rate, 
        row_len=row_len, num_rows_reported=num_rows_reported)
    sample_rate = 50E6/num_rows/data_rate/row_len
    wait_time = n_frames/sample_rate
    time.sleep(wait_time)
    stop_acq(S)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
    parser.add_argument('--slow-iv', action='store_true', default=False,
        help='Take IV curve using the slow method.')
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

    # Extract inputs
    args = parser.parse_args()

    offline = args.offline

    # Check for too many commands
    n_cmds = (args.log is not None) + args.tes_bias + args.slow_iv + \
        args.plc + args.tune + args.start_acq + args.stop_acq + \
        args.last_tune + (args.use_tune is not None) + args.overbias_tes + \
        args.bias_bump + args.soft_reset + args.make_runfile + args.setup + \
        args.flux_ramp_off + args.all_off + args.check_lock + args.status
    if n_cmds > 1:
        sys.exit(0)

    S = pysmurf.SmurfControl(cfg_file=os.path.join(os.path.dirname(__file__), 
        '..', 'cfg_files' , cfg_filename), smurf_cmd_mode=True, setup=False, 
        offline=offline)

    if args.log is not None:
        S.log(args.log)

    ### Tuning related commands ###
    if args.last_tune:
        S.log('Loading in last tuning')
        S.tune(last_tune=True, make_plot=args.tune_make_plot)

    if args.use_tune is not None:
        S.log('Loading old tune from file: {}'.format(args.use_tune))
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
                bias_voltage_array = np.zeros((8,)) # hard-coded number of bias groups
                bias_voltage_array[S.all_groups] = bias_voltage # all_groups from cfg
            S.set_tes_bias_bipolar_array(bias_voltage_array, write_log=True)
        else:
            S.set_tes_bias_bipolar(args.bias_group, bias_voltage, 
                write_log=True)

    if args.overbias_tes:

        if S.high_current_mode_bool: # best thing I could think of, sorry -CY
            tes_bias=2. # drop down to 2V to wait
        else:
            tes_bias=19.9 # factor of 10ish from above

        if args.bias_group < 0:
            S.overbias_tes_all(overbias_wait=args.overbias_tes_wait, bias_groups=S.all_groups, 
                high_current_mode=S.high_current_mode_bool, tes_bias=tes_bias)
        else:
            S.overbias_tes(args.bias_group, overbias_wait=args.overbias_tes_wait, 
                high_current_mode=S.high_current_mode_bool, tes_bias=tes_bias)

    if args.slow_iv:
        iv_bias_high = args.iv_bias_high
        iv_bias_low = args.iv_bias_low
        iv_bias_step = args.iv_bias_step

        S.log('bias high {}'.format(iv_bias_high))
        S.log('bias low {}'.format(iv_bias_low))
        S.log('bias step {}'.format(iv_bias_step))
        # 20181223: CY took out IV biases in terms of current. Revert if you 
        #  decide you want it back

        if iv_bias_high > 19.9:
            iv_bias_high = 19.9

        if iv_bias_step < 0:
            iv_bias_step = np.abs(iv_bias_step)

        if args.bias_group < 0: # all
            S.log('running slow IV on all bias groups')
            S.slow_iv_all(bias_groups=S.all_groups, wait_time=args.iv_wait_time,
                bias_high=iv_bias_high, bias_low = iv_bias_low,
                high_current_wait=args.iv_high_current_wait,
                high_current_mode=S.high_current_mode_bool,
                bias_step=iv_bias_step, make_plot=False)
        else: # individual bias group
            S.log('running slow IV on bias group {}'.format(args.bias_group))
            S.slow_iv_all(bias_groups=np.array([args.bias_group]), 
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
        S.log('plc bias high {}'.format(bias_high))
        S.log('plc bias low {}'.format(S.get_tes_bias_bipolar_array()))
        S.log('plc bias step {}'.format(args.iv_bias_step))

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
            start_acq(S, args.num_rows, args.num_rows_reported,
                args.data_rate, args.row_len)
            make_runfile(S.output_dir, num_rows=args.num_rows,
                data_rate=args.data_rate, row_len=args.row_len,
                num_rows_reported=args.num_rows_reported)
            # why are we making a runfile though? do we intend to dump it? 

    if args.stop_acq:
        stop_acq(S)

    if args.soft_reset:
        S.log('Soft resetting')
        S.set_smurf_to_gcp_clear(True)
        time.sleep(.1) # make this longer, maybe? just a thought. it lasts ~15s in MCE
        S.set_smurf_to_gcp_clear(False)

    if args.setup:
        S.log('Running setup')
        S.setup()
        # anything we need to do with setting up streaming?

    if args.make_runfile:
        make_runfile(S.output_dir, num_rows=args.num_rows,
            data_rate=args.data_rate, row_len=args.row_len,
            num_rows_reported=args.num_rows_reported)
