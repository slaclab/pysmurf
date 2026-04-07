import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as cbar


def read_file(filepath, filename):
    readin = np.load(os.path.join(filepath, filename), allow_pickle=True)
    return readin['data'].item()

def write_file(data, filepath, filename=None, verbose=True):
    filename = data["series"] + "_" + filename
    if verbose: print("Storing data as: ", os.path.join(filepath, filename))
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    np.savez(os.path.join(filepath, filename), data=data)
    return filepath, filename

def show_metadata(data):
    print('-'*20)
    print('  ' + data['series'])
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(key + ': array of length ' + str(data[key].shape))
        else:
            print(key + ': ' + str(data[key]))
    print('-'*20)
    return

def unwrap_phases(data, force_line_delay_val=None, verbose=True):
    ## Set up output containers, unwrap phases (i.e. remove 2pi jumps)
    corrected_phases = np.zeros(len(data["phases"]-1))
    unwrapped = np.unwrap(data["phases"])
    ## Give user the option to manually set a line delay
    ## If no value is supplied, calculate the line delay from the data
    if force_line_delay_val is None:
        line_delay = np.mean(unwrapped[1:]-unwrapped[:-1])/(data["freqs"][1:]-data["freqs"][:-1])
        line_delay = np.mean(line_delay)
        if verbose: print("Calculated line delay:", line_delay)
    else:
         if verbose: print("Manually set line delay:", force_line_delay_val)
         line_delay = force_line_delay_val
    for n, phase in enumerate(unwrapped):
            corrected_phases[n] = phase - (data["freqs"][n] - data["freqs"][0])*line_delay
    return corrected_phases, line_delay


def get_powers_in_set(set_filepath):
    powers_list = []
    os.chdir(set_filepath)
    for dir in os.listdir():
        if dir[-4:] != '_dBm':
            print('Skipping ' + dir + ' because it doesn\'t follow convention of ending in _dBm')
            continue
        pwr_str = dir[:-4]
        try:
            pwr = int(pwr_str)
            powers_list.append(pwr)
        except:
            print('Failed to convert ' + pwr_str + ' to a float')
            continue
    return powers_list

def find_highest_power(set_filepath):
    powers_list = get_powers_in_set(set_filepath)
    return max(powers_list)

def fit_line_delay_of_highest_power(set_filepath, plot=True):
    highest_power = find_highest_power(set_filepath)
    highest_power_filepath = os.path.join(set_filepath, str(highest_power) + '_dBm')
    unwrapped_phases = []
    line_delays = []
    for file in os.listdir(highest_power_filepath):
        if file[-4:] == '.npz':
            data = read_file(filepath=highest_power_filepath, filename=file)
            unwrapped_phase, line_delay = unwrap_phases(data)
            line_delays.append(line_delay)
            unwrapped_phases.append(unwrapped_phase)
    avg_delay = np.mean(line_delays)
    if plot:
        for phases in unwrapped_phases:
            plt.plot(data["freqs"], phases)
        plt.title(f'Unwrapped phase of highest power: {highest_power} dBm')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.show()
    print('Found average line delay of', avg_delay)
    if len(line_delays) > 1:
        print('with standard deviation of', np.std(line_delays))
    return avg_delay

## Processes a powerscan file set
def unwrap_and_avg_set(set_filepath, force_line_delay=None, plot_line_delay=True):
    ## First use the highest power to determine the line delay
    avg_delay = fit_line_delay_of_highest_power(set_filepath, plot=plot_line_delay)
    ## Now use this delay to unwrap the phases of all the other powers
    for pwr_idx, power in enumerate(get_powers_in_set(set_filepath)):
        os.chdir(os.path.join(set_filepath, str(power) + '_dBm'))
        total_avgs_at_power = 0

        ## Iterate through each data file in the power folder
        for i, fn in enumerate(os.listdir()):
            if fn[-4:] != '.npz': continue
            f = read_file(filepath='./', filename=fn)

            ## Account for bug in old data collection
            if isinstance(f['averages'], list):
                f['averages'] = f['averages'][pwr_idx]

            ## Check for consistency in metadata
            if i == 0:
                freqs = f['freqs']
                amps = f['averages'] * f['amps']
                phases = f['averages'] * f['phases']
                unwrapped_phase, _ = unwrap_phases(f, force_line_delay_val=avg_delay, verbose=False)
                unwrapped_phases = f['averages'] * unwrapped_phase
                vna_power = f['vna_power']
                power_at_device = f['power_at_device']
                ifbw = f['bandwidth']
                amps = np.zeros(len(f['amps']))
                phases = np.zeros(len(f['phases']))
            else:
                if f['freqs'][0] != freqs[0] or f['freqs'][-1] != freqs[-1] or len(f['freqs']) != len(freqs):
                    print('Error: frequency mismatch! in file:', fn)
                    return
                if f['vna_power'] != vna_power:
                    print('Error: VNA power mismatch! in file:', fn)
                    return
                if f['power_at_device'] != power_at_device:
                    print('Error: power at device mismatch! in file:', fn)
                    return
                if f['bandwidth'] != ifbw:
                    print('Error: IF bandwidth mismatch! in file:', fn)
                    return          

            ## Average the data
            total_avgs_at_power += f['averages']
            amps += f['averages'] * f['amps']
            phases += f['averages'] * f['phases']
            unwrapped_phase, _ = unwrap_phases(f, force_line_delay_val=avg_delay, verbose=False)
            unwrapped_phases += f['averages'] * unwrapped_phase
            
        ## Normalize the data
        amps /= total_avgs_at_power
        phases /= total_avgs_at_power
        unwrapped_phases /= total_avgs_at_power

        ## Save the averaged data
        f['amps'] = amps
        f['phases'] = phases
        f['unwrapped_phases'] = unwrapped_phases
        f['series'] = 'averaged'
        f['averages'] = total_avgs_at_power
        write_file(data=f, filepath=os.path.join(set_filepath, 'Processed'), filename=(set_filepath[-15:]+f'_pwr_{power}'))
    return


## Sorts the processed files by power
def sort_processed_filelist(processed_set_filepath):
    filelist = os.listdir(processed_set_filepath)
    pwrs = []
    for f in filelist:
        if f[-4:] != '.npz': continue
        pwrs.append(int(f[29:-4]))
    sorted_filelist = [filelist[i] for i in np.argsort(pwrs)]
    return sorted_filelist


## --------------------------------
## Powerscan functions

## Plot traces of the powerscan data
def plot_powerscan_traces(amps_data, phases_data, powers, freqs, plot_title=None, 
                   power_start=None, power_end=None, 
                   freq_start=None, freq_end=None, savepath=None):
    ## Truncate data if necessary
    if power_start is not None or power_end is not None:
        power_start_idx = np.argmin(np.abs(powers - power_start))
        amps_data = amps_data[power_start_idx:]
        phases_data = phases_data[power_start_idx:]
        powers = powers[power_start_idx:]
    if power_end is not None:
        power_end_idx = np.argmin(np.abs(powers - power_end))
        amps_data = amps_data[:power_end_idx+1]
        phases_data = phases_data[:power_end_idx+1]
        powers = powers[:power_end_idx+1]
    if freq_start is not None:
        freq_start_idx = np.argmin(np.abs(freqs/1e6 - freq_start))
        freqs = freqs[freq_start_idx:]
        amps_data = amps_data[:, freq_start_idx:]
        phases_data = phases_data[:, freq_start_idx:]
    if freq_end is not None:
        freq_end_idx = np.argmin(np.abs(freqs/1e6 - freq_end)/1e6)
        freqs = freqs[:freq_end_idx+1]
        amps_data = amps_data[:, :freq_end_idx+1]
        phases_data = phases_data[:, :freq_end_idx+1]

    ## Make the immshow plots
    fig, ax = plt.subplots(figsize=(9,5))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(powers)))
    for i in range(len(powers)):
        ax.plot(freqs/1e6, amps_data[i], color=colors[i])
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=min(powers), vmax=max(powers)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Power [dB]')
    plt.ylabel('S21 Phase [dBm]')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('S21 Amplitude [dBm]')
    plt.title(plot_title)
    ## Save the plot
    if savepath is not None:
        os.chdir(savepath)
        filename = plot_title + '_amp_traces_00'
        index = 00
        while os.path.exists(filename + '.png'):
            index += 1
            filename = filename[:-2] + str(index)
        plt.savefig(filename + '.png')

    fig, ax = plt.subplots(figsize=(9,5))
    for i in range(len(powers)):
        ax.plot(freqs/1e6, phases_data[i], color=colors[i])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Power [dB]')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('S21 Phase [dBm]')
    plt.title(plot_title)
    ## Save the plot
    if savepath is not None:
        os.chdir(savepath)
        filename = plot_title + '_phase_traces_00'
        index = 00
        while os.path.exists(filename + '.png'):
            index += 1
            filename = filename[:-2] + str(index)
        plt.savefig(filename + '.png')
    return


## Plot a heatmap of the powerscan data
def plot_powerscan_heatmap(amps_data, phases_data, powers, freqs, plot_title=None, 
                   power_start=None, power_end=None, 
                   freq_start=None, freq_end=None, savepath=None):
    ## Truncate data if necessary
    if power_start is not None or power_end is not None:
        power_start_idx = np.argmin(np.abs(powers - power_start))
        amps_data = amps_data[power_start_idx:]
        phases_data = phases_data[power_start_idx:]
        powers = powers[power_start_idx:]
    if power_end is not None:
        power_end_idx = np.argmin(np.abs(powers - power_end))
        amps_data = amps_data[:power_end_idx+1]
        phases_data = phases_data[:power_end_idx+1]
        powers = powers[:power_end_idx+1]
    if freq_start is not None:
        freq_start_idx = np.argmin(np.abs(freqs/1e6 - freq_start))
        freqs = freqs[freq_start_idx:]
        amps_data = amps_data[:, freq_start_idx:]
        phases_data = phases_data[:, freq_start_idx:]
    if freq_end is not None:
        freq_end_idx = np.argmin(np.abs(freqs/1e6 - freq_end))
        freqs = freqs[:freq_end_idx+1]
        amps_data = amps_data[:, :freq_end_idx+1]
        phases_data = phases_data[:, :freq_end_idx+1]

    ## Make the immshow plots
    fig, ax = plt.subplots()
    im = ax.imshow(amps_data, aspect='auto', extent=[freqs[0]/1e6, freqs[-1]/1e6, powers[0], powers[-1]], origin='lower', cmap = 'magma')
    plt.colorbar(im)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power [dBm]')
    plt.title("S21 Amplitude\n"+plot_title)
    ## Make sure not to overwrite the previous plot
    if savepath is not None:
        os.chdir(savepath)
        filename = plot_title + '_amp_heatmap_00'
        index = 00
        while os.path.exists(filename + '.png'):
            index += 1
            filename = filename[:-2] + str(index)
        plt.savefig(filename + '.png')

    fig, ax = plt.subplots()
    im = ax.imshow(phases_data, aspect='auto', extent=[freqs[0]/1e6, freqs[-1]/1e6, powers[0], powers[-1]], origin='lower', cmap = 'magma')
    plt.colorbar(im)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power [dBm]')
    plt.title("S21 Phase\n"+plot_title)
    ## Make sure not to overwrite the previous plot
    if savepath is not None:
        os.chdir(savepath)
        filename = plot_title + '_phase_heatmap_00'
        index = 00
        while os.path.exists(filename + '.png'):
            index += 1
            filename = filename[:-2] + str(index)
        plt.savefig(filename + '.png')
    return

def series_list_to_elapsed_time(series_list):
    elapsed_times = np.zeros(len(series_list))
    series = series_list[0]
    start_time = datetime.datetime(int(series[:4]), int(series[4:6]), 
                                int(series[6:8]), int(series[9:11]), 
                                int(series[11:13]), int(series[13:15]))
    for i, series in enumerate(series_list):
        time = datetime.datetime(int(series[:4]), int(series[4:6]), 
                                int(series[6:8]), int(series[9:11]), 
                                int(series[11:13]), int(series[13:15]))
        elapsed_times[i] = (time - start_time).total_seconds()
    return elapsed_times



## --------------------------------
## Navigate long timestream datastructure and import fscans
def pull_freqscan_data(data_date, global_datapath):
    """
    Pulls in all freqscan data from a given data_date.  
    Returns a dataframe of amplitudes and phases, as well as the frequency array.
    ---------
    Args:
    data_date (str): Date of the data to be analyzed
    global_datapath (str): Path to the data directory
    ---------
    Returns:
    fscan_amps (np.array): Array of amplitudes from the freqscan
    fscan_phases (np.array): Array of phases from the freqscan
    freqs (np.array): Array of frequencies from the freqscan
    """
    ## Set names are subdirectories of the data directory
    set_dirs = [os.path.join(global_datapath, data_date, item) for item in os.listdir(os.path.join(global_datapath, data_date))]
    set_dirs = [item for item in set_dirs if os.path.isdir(item) and 'plots' not in item and 'processed' not in item]
    processed_dir = os.path.join(global_datapath, data_date, 'processed')

    ## Go through list, import fsans only.  Put into dataframe
    fscan_amps = []
    fscan_phases = []
    fscan_sers = []

    idx = 0
    for set_dir in set_dirs:
        print(f'Processing {set_dir} ({idx+1}/{len(set_dirs)})')

        ## Look for freqscan in the directory
        for item in os.listdir(set_dir):
            if 'fscan' in item:
                fscan_file = item
                break
        else:
            print(f'No freqscan found in {set_dir}\n')
            continue

        fscan = read_file(filepath=set_dir, filename=fscan_file)
        if idx==0:
            fscan_ref = fscan
            freqs = fscan['freqs']
        fscan_amps.append(fscan['amps'])
        fscan_phases.append(fscan['phases'])
        fscan_sers.append(fscan['series'])
        idx += 1
    
    fscan_amps = np.array(fscan_amps)
    fscan_phases = np.array(fscan_phases)
    fscan_times = series_list_to_elapsed_time(fscan_sers)
    title_str = f"{data_date}_{fscan['device']}{fscan['qubit']}_pwr{fscan['vna_power']}_ifbw{fscan['bandwidth']}"

    ## Save to processed directory
    if not os.path.isdir(processed_dir): os.mkdir(processed_dir)
    np.save(os.path.join(processed_dir, f'{title_str}_fscan_amps.npy'), fscan_amps)
    np.save(os.path.join(processed_dir, f'{title_str}_fscan_phases.npy'), fscan_phases)
    np.save(os.path.join(processed_dir, f'{title_str}_freqs.npy'), freqs)
    np.save(os.path.join(processed_dir, f'{title_str}_fscan_times.npy'), fscan_times)
    print(f'Saved to {processed_dir}')
    return fscan_amps, fscan_phases, freqs, fscan_times