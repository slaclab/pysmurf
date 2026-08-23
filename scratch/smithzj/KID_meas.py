import numpy as np
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pylab as plt
import pickle 
from scipy.signal import butter, welch, filtfilt, periodogram, savgol_filter
import os
import json
os.chdir('/usr/local/src/pysmurf/scratch/smithzj/')
from ResonanceFitter import *
def welch_IQ(iq, fs,  plot=True, welch_nperseg=2**18, title='', show_plot=True):
    i = iq.real
    q = iq.imag
    ffi, pxxi = welch(i,fs=fs, nperseg=welch_nperseg)
    ffq, pxxq = welch(q,fs=fs, nperseg=welch_nperseg)

    # scale to dBc/Hz by the voltage magnitude
    magfac = np.mean(q)**2 + np.mean(i)**2
    pxxi_dbc = 10. * np.log10(pxxi / magfac)
    pxxq_dbc = 10. * np.log10(pxxq / magfac)


    if plot:
        plt.gca().semilogx(ffi,pxxi_dbc,linestyle='-',label=f'i {title}')
        plt.gca().semilogx(ffq,pxxq_dbc, linestyle='--',label=f'q {title}')
        plt.ylabel('dBc/Hz')
        plt.xlabel('Frequency (Hz)')
        plt.title(title)
        plt.legend(loc='lower left')
        if show_plot: plt.show()
    return
    
def takeDebugData(S, band, channel, nsamp, plot=True, welch_nperseg=2**18, show_plot=True):
    timestamp = S.get_timestamp() 
    filename = f'{timestamp}_single_channel_b{band}ch{channel:03}'
    i,q,sync = S.take_debug_data(band=band,channel=channel,rf_iq=True,nsamp=nsamp,filename = filename) 
    i = i / (1.2)
    q = q / (-1.2)
    iq = i + 1j * q
    if plot: 
        fs = S.get_channel_frequency_mhz(band) * 1.0E6
        welch_IQ(iq, fs, welch_nperseg, title=filename, show_plot=show_plot)
        ### save plots 
        fig = plt.gcf()
        plt.savefig(f'/data/smurf_data/mkid_1tone_streaming_metadata/_Figs/{filename}_psd', fmt='tiff')
        plt.show()
    return iq, sync, filename
def rotate_to_ideal_compact(tune_dict, fopt):  
    # osmond
    z = tune_dict['r']
    f = tune_dict['freqs']
    fr = fopt['f0']
    a =  fopt['zOff']
    tau =  fopt['tau']
    phi = fopt['phi']
    return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))
def rotate_to_ideal_semi_compact(z, f, fopt):  
    # osmond
    fr = fopt['f0']
    a =  fopt['zOff']
    tau =  fopt['tau']
    phi = fopt['phi']
    return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))
def saveDebugData(S, band, channel, iq, sync, filename):
    """
    run this immediatley after taking data, to store data taking 
    don't run independently! Should only be run inside takeIQ
    nsamp= sample rate, as used by takeIQ
    data_file_path= complete file path for where associated takeIQ .dat file
    file_code: (string) this is up to you! whatever might help you understand the setup.
    """
    print('getting parameters...') 
    freq_in_Hz = get_freq_in_Hz(S, band=band, channel=channel)
    eta_phase_degree = S.get_eta_phase_degree_channel(band,channel)
    eta_phase_rad = np.deg2rad(eta_phase_degree)
    fs = S.get_channel_frequency_mhz(band) * 1.0E6 #For this mode of smurf, channel_freq is sample rate
    att_uc = S.get_att_uc(band)
    att_dc = S.get_att_dc(band)
    
    amplitude_scale_array = S.get_amplitude_scale_array(band).tolist()

    data_dict = {}
    data_dict['band'] = band
    data_dict['channel'] = channel
    data_dict['iq'] = iq
    # data_dict['sync'] = sync
    data_dict['nsamp'] = len(iq.real)
    data_dict['freq_in_Hz'] = freq_in_Hz
    data_dict['eta_phase_rad'] = eta_phase_rad
    data_dict['fs'] = fs
    data_dict['amplitude_scale_array'] = amplitude_scale_array
    data_dict['att_uc'] = att_uc
    data_dict['att_dc'] = att_dc
    
    data_file_name = f'{os.path.basename(filename)}'
    full_metadata_path = os.path.join('/data/smurf_data/mkid_1tone_streaming_metadata/_Data/', data_file_name)+'.pkl'
    print(f"saving metatdata to {full_metadata_path} ")
    
    with open(full_metadata_path, 'wb') as f:
        pickle.dump(data_dict, f)
        #np.save(full_metadata_path, attributes_dict)
    return full_metadata_path


def loadDebugData(full_data_path):
    with open(full_data_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

    return

def get_freq_in_Hz(S, band, channel, print_out=False):
    sb = S.get_subband_from_channel(band,channel)
    fsbc = S.get_subband_centers(band)[1][sb]
    fbc = S.get_band_center_mhz(band)
    fcc = S.get_center_frequency_mhz_channel(band,channel)
    if print_out==True:
        print(f'sb={sb} sbc={fsbc} bc={fbc}')
        print(f'fr={(fsbc+fbc+fcc)} MHz')
    freq_in_Hz = (fsbc+fbc+fcc)*1e6
    return freq_in_Hz 


def slowInitializeMkidSmurf(S,band,tone_power,ff_freq_min=-250,ff_freq_max=250, amp_cut=0.05, grad_descent_averages=100, eta_scan_averages=1000):
    """
    use this to find your resonant frequency. 
        
    uc_att is up converter attenuation (max 0) 
        
    ends with all tones off. 
    """
    S.find_freq(band,tone_power=tone_power,\
                start_freq=ff_freq_min,stop_freq=ff_freq_max,make_plot=True,show_plot=True,\
                amp_cut=amp_cut,rolling_med=False) #amp cut decides what you count for a resonator 
    S.setup_notches(band,tone_power=tone_power, new_master_assignment=True)
    S.plot_tune_summary(band,eta_scan=True,show_plot=True)
    print(f'S.which_on({band})={S.which_on(band)}')
    resonant_channel = S.which_on(band)[0] #int(input("(int) self.resonant_channel="))
    for ch in S.which_on(band):
        print('turning off non resonant channels...')
        if ch != resonant_channel:
            S.channel_off(band, channel=int(ch))
    print(f'S.which_on({band})={S.which_on(band)}')
    ff_freq = get_freq_in_Hz(S, band, resonant_channel, print_out=True)
    S.set_gradient_descent_averages(band,grad_descent_averages)   #get to minimum of the transmission
    S.run_serial_gradient_descent(band)  #run it on every tone that is on 
    S.set_eta_scan_averages(band, eta_scan_averages)  #set averages high
    S.run_serial_eta_scan(band)   #try to measure line and rotate 
    print(f'S.get_eta_mag_scaled_channel={S.get_eta_mag_scaled_channel(band,resonant_channel)}')
    print(f'S.get_eta_phase_degree_channel={S.get_eta_phase_degree_channel(band,resonant_channel)}')
    ff_freq = get_freq_in_Hz(S,band, resonant_channel, print_out=True)
    S.set_feedback_enable_channel(band,resonant_channel,0) #stop feedback on resonant frequency. Just send 1 fixed tone
    
def load_tune_file(tune_file, band, db_offset=1):
    """
    tune_file: (str) the full file path to tune file
    band: (int) what band was data taken on? 
    returns dict of file name, iq scan, freqs, and res freq  
    """
    tf =  np.load(tune_file, allow_pickle=True).item()
    r = tf[band]['resonances'][0]['resp_eta_scan']
    freqs = tf[band]['resonances'][0]['freq_eta_scan']
    fr = tf[band]['resonances'][0]['freq']
    tune_dict = {}
    tune_dict['fname'] = tune_file
    tune_dict['r'] = r * db_offset
    tune_dict['freqs'] = freqs
    tune_dict['fr'] = fr
    return tune_dict


def resonator_basis(iq_ideal_basis, fine_pars, axis=-1):
    dS21 = iq_ideal_basis - np.mean(iq_ideal_basis,dtype=complex, axis=axis, keepdims=True)
    frequency =  dS21.imag * fine_pars['Qc'] / ( 2 * fine_pars['Qr'] **2)
    dissipation = dS21.real * fine_pars['Qc']  / (fine_pars['Qr'] **2)
    return frequency, dissipation

def fit_tune_file(tune_dict, plot=True):
    if tune_dict['fr'] < max(tune_dict['freqs']) and tune_dict['fr'] > min(tune_dict['freqs']):
        i_fr_0 = np.argmin(abs(tune_dict['freqs']-tune_dict['fr']))
        i_start = min([0, i_fr_0-1500])
        i_end  = max([i_fr_0+1500, len(tune_dict['freqs'])])

    fn = tune_dict['fname']+'_fit'
    
    fine_pars, fine_errs = finefit(tune_dict['freqs'][i_start:i_end], tune_dict['r'][i_start:i_end], tune_dict['fr'], fn, show_plots=plot)
    print(f'fine_pars: {fine_pars}')
    #print(f'fine_errs: {fine_errs}')
    print('Qi:', fine_pars['Qr']*( 1-fine_pars['Qr'] / fine_pars['Qc'] ) )
    return fine_pars, fine_errs





"""
    Fits data from tune file and plots fit and tune file on same plot. Returns fit parameters and errors.
    
"""
def fit_data(tune_file, db_offset=1):
     
    tune_dict=load_tune_file(tune_file, band, db_offset)
     
    ##fit_tune_file returns two dictionaries, one with fit values and the other with errors. In each the keys are:
        ## f0: the resonant frequency
        ## Qr
        ## phi
        ## zoff (this is "a" in the usual equation form ie ae^{2piitau}....)
        ## QcHat
        ## tau
        ## Qc

    fit_dict, fine_errs = fit_tune_file(tune_dict, plot=True)
    print("Fit dictionary parameters:  ", fit_dict)

    plot_fit(tune_dict, fit_dict)
    
    return fit_dict, fine_errs

"""
    Plots the S21 equation with tune file and fit dictionary parameters. 
    
"""

def plot_fit(tune_dict, fit_dict):

    ##This is the fit parameters plugged int othe fit function.
    iq_fit = resfunc3(tune_dict['freqs'], fit_dict['f0'], fit_dict['Qr'], fit_dict['QcHat'], fit_dict['zOff'], fit_dict['phi'],fit_dict['tau'])

    ##This is also the fit parameters plugged into the fit function, but the x data is only the resonant frequency (so returns one point).
    iq_res = resfunc3(fit_dict['f0'], fit_dict['f0'], fit_dict['Qr'], fit_dict['QcHat'], fit_dict['zOff'], fit_dict['phi'],fit_dict['tau'])

    ## This is the data rotated to the ideal basis
    VNA_ideal_basis = rotate_to_ideal_compact(tune_dict, fit_dict)
    ## This is the fit rotated to the ideal basis.
    fit_ideal_basis = rotate_to_ideal_semi_compact(iq_fit, tune_dict['freqs'], fit_dict)
    ## This is the resonance from the fit rotated to the ideal basis.
    iq_res_rotated = rotate_to_ideal_semi_compact(iq_res,  fit_dict['f0'], fit_dict)

    ## This is the frequency and dissipation basis values, from both the fit and the data.
    VNA_freq, VNA_diss = resonator_basis(VNA_ideal_basis, fit_dict, axis=None)
    fit_freq, fit_diss = resonator_basis(fit_ideal_basis, fit_dict, axis=None)
    iq_res_freq, iq_res_diss = resonator_basis(iq_res_rotated, fit_dict, axis=None)

    fig,axs = plt.subplots(ncols=2, figsize=(10, 10) )#sharex=True, sharey=True)

    print("Plotting iq, fit, and fr in ideal basis")
    axs[0].plot(tune_dict['r'].real, tune_dict['r'].imag,'C0.-', label='VNA iq')
    axs[0].plot(iq_fit.real, iq_fit.imag,'C1-', label='fit')
    axs[0].plot(iq_res.real, iq_res.imag, 'ro', label='fr from fit')
    axs[0].legend()

    print("Plotting iq, fit, and fr in resonator basis")
    axs[1].plot(VNA_ideal_basis.real, VNA_ideal_basis.imag,'C0.-', label='VNA iq in ideal basis')
    axs[1].plot(fit_ideal_basis.real, fit_ideal_basis.imag,'C1-', label='rotated fit')
    axs[1].plot(iq_res_rotated.real, iq_res_rotated.imag, 'ro', label='rotated fr from fit')
    
    
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].set_ylabel("Q, unrotated")
    axs[1].set_ylabel("Q, rotated")
    axs[0].set_xlabel("I, unrotated")
    axs[1].set_xlabel("I, rotated")
    
    return None

"""
    Takes data using debug_data. Saves it to a file. Returns data (iq), sync, and fthe filename. 
    
"""
def takeDebugData_new(S, band, channel, nsamp, plot=True, welch_nperseg=2**18, show_plot=True, channel_mode=0):
    timestamp = S.get_timestamp() 
    filename = f'{timestamp}_single_channel_b{band}ch{channel:03}'
    i,q,sync = S.take_debug_data(band=band,channel=channel,rf_iq=True,nsamp=nsamp,filename=filename, single_channel_readout=channel_mode) 
    i = i / (1.2)
    q = q / (-1.2)
    iq = i + 1j * q
    if plot: 
        fs = S.get_channel_frequency_mhz(band) * 1.0E6
        welch_IQ(iq, fs, welch_nperseg, title=filename, show_plot=show_plot)
        ### save plots 
        fig = plt.gcf()
        plt.savefig(f'/data/smurf_data/mkid_1tone_streaming_metadata/_Figs/{filename}_psd', fmt='tiff')
        plt.show()
    return iq, sync, filename

"""
    Saves the debug data to the log file. 
    
"""

def logDebugData(S, band, channel, iq, sync, filename, tunefile, cooldown_str, char_avgs={}, awg_settings={}, led_settings={}, MEMS_settings={}):
    freq_in_Hz = get_freq_in_Hz(S, band=band, channel=channel)
    eta_phase_degree = S.get_eta_phase_degree_channel(band,channel)
    eta_phase_rad = np.deg2rad(eta_phase_degree)
    fs = S.get_channel_frequency_mhz(band) * 1.0E6 #For this mode of smurf, channel_freq is sample rate
    att_uc = S.get_att_uc(band)
    att_dc = S.get_att_dc(band)    

            
    amplitude_scale_array = S.get_amplitude_scale_array(band).tolist()
    channels = []
    for i in range(len(amplitude_scale_array)):
        if amplitude_scale_array[i] != 0:
            channels.append(i)
            

    data_dict = {}
    data_dict['band'] = band
    data_dict['channels'] = channels
    data_dict['filename'] = filename
    data_dict['nsamp'] = len(iq.real)
    data_dict['freq_in_Hz'] = freq_in_Hz
    data_dict['eta_phase_rad'] = eta_phase_rad.tolist()
    data_dict['fs'] = fs
    data_dict['amplitude_scale_array'] = amplitude_scale_array
    data_dict['att_uc'] = att_uc
    data_dict['att_dc'] = att_dc
    data_dict['tunefile']=tunefile
    data_dict['awg_settings']=awg_settings
    data_dict['char_avgs']=char_avgs
    data_dict['cooldown_str']= cooldown_str
    data_dict['MEMS'] = MEMS_settings
    data_dict['led_settings'] = led_settings

    S.log("JSON DICT FOR " + filename)
    S.log(json.dumps(data_dict))

"""
    Loads the debug data from the logfile into a dictionary. Returns the dictionary.
"""
def loadDebugData(logfile, filename):
    read_log = open(logfile, 'r')
    done =False
    found=False
    for l_no, line in enumerate(read_log):
        if found and not done:
            text_dict=line
            done = True
        elif "JSON DICT FOR " + filename in line:
            found=True
    startIndex = text_dict.find("{")
    data_dict = json.loads(text_dict[startIndex:])
    return data_dict