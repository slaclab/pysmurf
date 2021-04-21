#!/usr/bin/env python3
import argparse
import time
import pysmurf.client
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import epics
import os

def run(band, epics_root=None, config_file=None):
    '''
    Run the loopback test.
    '''
    # Make storage dictionary
    status = {}
    status['band'] = band

    def execute(status_dict, func, label, save_dict=True):
        '''
        Run and time any arbitrary pysmurf function.
        Must pass function as a lambda.
        Taken from scratch/eyy/test_scripts/profile_band.py.

        Arguments
        -----
        status_dict : dict
            The dictionary that stores all the start/end times and outputs of
            the functions.
        label : str
            The descriptor to label the function.
        save_dict : bool
            Whether to save the dictionary. Default True.

        Returns
        -----
        status_dict : dict
            The updated status dictionary.
        '''
        # Add label to dict
        status_dict[label] = {}

        # Note start time
        status_dict[label]['start'] = S.get_timestamp(as_int=True)

        # Run function
        status_dict[label]['output'] = func()

        # Add end time
        status_dict[label]['end'] = S.get_timestamp(as_int=True)

        # Save the dictionary
        np.save(os.path.join(S.output_dir, 'status'),
                status_dict)

        return status_dict

    # Define variables
    # Attenuator values to loop through
    uc_vals = [0, 18]
    dc_vals = [0, 18]
    # The up and down converter attenuators are accounted for in the full band response result
    # So (resp with attenuators)/(resp without attenuators) should be within 1 +/- some small full_band_resp_ratio_sigma
    # See get_sigma_full_band_resp_ratios.py where I studied what full_band_resp_ratio_sigma should be
    full_band_resp_ratio_sigma = 0.01

    # Initialize
    if epics_root is None:
        epics_prefix = 'smurf_server_s5'
    if config_file is None:
        config_file  = '/usr/local/src/pysmurf/cfg_files/caltech/tkid/experiment_tkid_lbOnlyBay0.cfg'
    S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
                                    setup=False, make_logfile=True)

    # S.output_dir is e.g. '/data/smurf_data/20210415/1618444853/outputs'
    print('All outputs going to: ')
    print(S.output_dir)

    # Setup, should return True
    status = execute(status, lambda: S.setup(), 'setup')
    if not status['setup']['output']:
        status['setup']['pass'] = 'Fail'
        raise RuntimeError('Failed to setup. Aborting test.')
    else:
        status['setup']['pass'] = 'Pass'
        print('Setup successful.')

    # Estimate phase delay
    print('Estimating phase delay...')
    status = execute(status,
                     lambda: S.estimate_phase_delay(band, make_plot=True, save_plot=True,
                                                    show_plot=False),
                     'estimate_phase_delay')
    # Estimated processing delay should be close to 10, at least 1
    # Residual total delay should be close to 0
    pass_arr = []
    if status['estimate_phase_delay']['output'][0] < 1:
        delay = status['estimate_phase_delay']['output'][0]
        pass_arr.append(f'Fail: estimated processing delay is less than 1 second: {delay}')
        print(f'WARNING: estimated processing delay is less than 1 second: {delay}')
    if status['estimate_phase_delay']['output'][1] > 1e-4:
        delay = status['estimate_phase_delay']['output'][1]
        pass_arr.append(f'Fail: residual total delay is greater than 1e-4 seconds: {delay}')
        print(f'WARNING: residual total delay is greater than 1e-4 seconds: {delay}')
    if not pass_arr:
        pass_arr.append('Pass')
    status['estimate_phase_delay']['pass'] = '<br />'.join(pass_arr)

    # Loop through different attenuator values
    pass_arr =[]
    status['uc_vals'] = uc_vals
    status['dc_vals'] = dc_vals
    for uc in uc_vals:
        S.set_att_uc(band, uc)
        for dc in dc_vals:
            S.set_att_dc(band, dc)
            print(f'Attenuator values uc {uc} dc {dc}')

            # Full band response (output is a tuple with two arrays, frequency and response)
            print('Getting full band response...')
            status = execute(status,
                             lambda: S.full_band_resp(band, n_scan=5, make_plot=True, save_plot=True,
                                                      show_plot=False, return_plot_path=True),
                             f'full_band_resp (uc {uc}, dc {dc})')
            # Should return something reasonable, not just white noise
            if len(status[f'full_band_resp (uc {uc}, dc {dc})']['output'][1]) < 1:
                pass_arr.append(f'Fail: full band response output length 0 for uc {uc} dc {dc}!')
                print('WARNING: full band response output length 0!')
            elif uc == 0 and dc == 0:
                # Check to see if the attenuators are actually on
                if S.get_att_uc(band) != uc or S.get_att_dc(band) != dc:
                    pass_arr.append(f'Fail: attenuators not correctly set for uc {uc} dc {dc}!')
                    print('WARNING: attenuators not correctly set!')
            else:
                # Check to see if the attenuators are actually on
                if S.get_att_uc(band) != uc or S.get_att_dc(band) != dc:
                    pass_arr.append(f'Fail: attenuators not correctly set for uc {uc} dc {dc}!')
                    print('WARNING: attenuators not correctly set!')
                # The up and down converter attenuators are accounted for, so the response should be unaffected by uc and dc
                resp_0 = status['full_band_resp (uc 0, dc 0)']['output'][1]
                resp = status[f'full_band_resp (uc {uc}, dc {dc})']['output'][1]
                ratio = np.mean(resp/resp_0)
                if ratio > 1+full_band_resp_ratio_sigma or ratio < 1-full_band_resp_ratio_sigma:
                    print('WARNING: failed to correct for attenuators.')
                    pass_arr.append(f'Fail: failed to correct for attenuators for uc {uc} dc {dc}')

            # Find frequencies
            print('Finding frequencies...')
            status = execute(status,
                             lambda: S.find_freq(band=band,
                                                 start_freq=-250, stop_freq=250,
                                                 tone_power=10, make_plot=True, save_plot=True),
                             f'find_freq (uc {uc}, dc {dc})')
    if not pass_arr:
        pass_arr.append('Pass')
    status['full_band_resp_pass'] = '<br />'.join(pass_arr)

    # Make plot of all the full band response plots
    timestamp = S.get_timestamp()
    fig, ax = plt.subplots(1, figsize=(11, 6))
    for uc in uc_vals:
        for dc in dc_vals:
            f = status[f'full_band_resp (uc {uc}, dc {dc})']['output'][0]
            resp = status[f'full_band_resp (uc {uc}, dc {dc})']['output'][1]
            f_plot = f / 1.0e6
            plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(f_plot[plot_idx], np.log10(np.abs(resp[plot_idx])), color=color,
                    label=f'uc {uc}, dc {dc}')
    ax.set_xlabel('Freq [MHz]')
    ax.set_ylabel('Response')
    plt.title(f'full_band_resp for Multiple Attenuator Values')
    plt.tight_layout()
    plt.legend()
    plot_path = os.path.join(S.plot_dir, f'{timestamp}_b{band}_full_band_resp.png')
    plt.savefig(plot_path, bbox_inches='tight')
    status['full_band_resp_plot'] = f'{timestamp}_b{band}_full_band_resp.png'

    # Turn on a channel and stream data
    print('Try streaming data...')
    S.set_amplitude_scale_channel(band, 0, 11) # 11 or 12 for tone value
    S.set_feedback_gain(band=band, val=2048)
    S.set_feedback_limit_khz(band=band, feedback_limit_khz=100)
    status = execute(status,
                     lambda: S.take_debug_data(band=band, channel=0, nsamp=2**19,
                                               IQstream=0, single_channel_readout=2),
                     'take_debug_data')
    # Check that something has been written
    if len(status['take_debug_data']['output'][0]) != 2**19:
        status['take_debug_data']['pass'] = 'Fail'
        print('Failed to grab data.')
    else:
        status['take_debug_data']['pass'] = 'Pass'
        print('Successfully streamed data.')
    # Plot
    fig, ax = plt.subplots(2, figsize=(11, 11))
    f = status['take_debug_data']['output'][0]
    dF = status['take_debug_data']['output'][1]
    t = np.arange(len(f))
    t = t*1000/2.4e6
    ax[0].plot(t, dF)
    ax[0].set_ylabel('Frequency Error')
    ax[1].plot(t, f)
    ax[1].set_ylabel('Tracked Frequency')
    ax[1].set_xlabel('Time (uS)')
    plt.tight_layout()
    plot_path = os.path.join(S.plot_dir, f'b{band}_take_debug_data.png')
    plt.savefig(plot_path, bbox_inches='tight')
    status['take_debug_data']['plot'] = f'b{band}_take_debug_data.png'

    # Save the dictionary
    np.save(os.path.join(S.output_dir, 'status'), status)
    # Make webpage
    html_path = make_html(os.path.split(S.output_dir)[0])

    return html_path

def make_html(data_path):
    '''
    Makes the HTML page.

    Arguments:
    -----
    data_path : str
        The full path to the data output file.

    Returns
    -----
    html_path : str
        The path to the HTML output.
    '''
    import shutil
    import fileinput
    import datetime
    import glob
    import re

    # Make output directories
    script_path = os.path.dirname(os.path.realpath(__file__))
    template_path = os.path.join(script_path, 'page_template')
    html_path = os.path.join(data_path, 'loopback_test_summary')

    print(f'Making HTML output in: {html_path}')

    # Copy template directory
    shutil.copytree(template_path, html_path)

    # Load status dict
    status = np.load(os.path.join(data_path, 'outputs/status.npy'),
                     allow_pickle=True).item()
    band = status['band']

    def replace_str(filename, search_str, replace_str):
        with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(search_str, replace_str), end='')

    index_path = os.path.join(html_path, 'index.html')

    # Fill in time
    replace_str(index_path, '[[DATETIME]]',
                datetime.datetime.fromtimestamp(status['setup']['start']).strftime('%Y-%m-%d'))

    # Fill in band number
    replace_str(index_path, '[[BAND]]', str(band))

    # Summary
    setup = status['setup']['pass']
    estimate_time_delay = status['estimate_phase_delay']['pass']
    full_band_response = status['full_band_resp_pass']
    stream_data = status['take_debug_data']['pass']
    summary_str = '<table style=\"width:80%\" align=\"center\" border=\"1\">'
    summary_str += f'<tr><td>Band</td><td>{band}</td>'
    summary_str += f'<tr><td>Setup</td><td>{setup}</td>'
    summary_str += f'<tr><td>Estimate Time Delay</td><td>{estimate_time_delay}</td>'
    summary_str += f'<tr><td>Full Band Response</td><td>{full_band_response}</td>'
    summary_str += f'<tr><td>Stream Data and Read Back</td><td>{stream_data}</td>'
    summary_str += '</table>'
    replace_str(index_path, '[[SUMMARY]]',
                summary_str)

    # Do timing calculations
    skip_keys = ['band', 'full_band_resp_plot', 'full_band_resp_pass', 'uc_vals', 'dc_vals']
    timing_str = '<table style=\"width:60%\" align=\"center\" border=\"1\">'
    timing_str += '<tr><th>Function</th><th>Time [s]</th></tr>'
    for k in list(status.keys()):
        if k not in skip_keys:
            t = status[k]['end'] - status[k]['start']
            timing_str += f'<tr><td>{k}</td><td>{t}</td></tr>'
    timing_str += '</table>'
    replace_str(index_path, '[[TIMING]]',
                timing_str)

    # Add full band response plot
    basename = status['full_band_resp_plot']
    replace_str(index_path, '[[FULL_BAND_RESP]]',
                os.path.join('../plots/', basename))

    # Add plots for find_freq
    find_freq_timestamps = []
    uc_vals = status['uc_vals']
    dc_vals = status['dc_vals']
    fname_list = glob.glob(os.path.join(data_path,'plots/*_amp_sweep_0.png'))
    for fname in fname_list:
        match = re.search('(\d+)_.*.png', os.path.basename(fname))
        find_freq_timestamps.append(match.group(1))
    find_freq_list = ''
    for i, uc in enumerate(uc_vals):
        for j, dc in enumerate(dc_vals):
            find_freq_list = find_freq_list + "\'uc " + str(uc) + ", dc " + str(dc) + "|" + find_freq_timestamps[i*len(dc_vals)+j] + "\', "
    find_freq_list = '[' + find_freq_list + ']'
    replace_str(index_path, '[[FIND_FREQ_LIST]]', find_freq_list)
    instr = f"p[\'uc_dc\'] + \'_amp_sweep_0.png\'"
    replace_str(index_path, '[[FIND_FREQ_PATH]]', instr)
    replace_str(index_path, '[[UC_DC_0]]', find_freq_timestamps[0])

    # Add plot of the streamed data
    basename = status['take_debug_data']['plot']
    debug_data_loc = os.path.join('../plots/', basename)
    replace_str(index_path, '[[TAKE_DEBUG_DATA]]', debug_data_loc)

    return html_path

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=int, required=True,
                        help="The band to run the analysis on.")
    parser.add_argument("--epics-root", type=str, required=False,
                        help="The epics root.")
    parser.add_argument("--config-file", type=str, required=False,
                        help="The configuration file to use for this test.")

    # Parse arguments
    args = parser.parse_args()

    # Run the generator script
    run(args.band, args.epics_root, args.config_file)
