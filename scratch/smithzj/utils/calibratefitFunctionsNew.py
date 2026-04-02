import numpy as np
from ResonanceFitter import *
from data_collection_utils import *

import pysmurf.client

######################################################################################################
#################################### functions for tune file #########################################
######################################################################################################
"""
def loadTuneFile(tune_file, band,db_offset):

    #tune_file: (str) the full file path to tune file
    #band: (int) what band was data taken on? 
    #returns dict of file name, iq scan, freqs, and res freq  

    tf =  np.load(tune_file, allow_pickle=True).item()
    ## This is the response - so the transmission for I and Q
    r = tf[band]['resonances'][0]['resp_eta_scan']
    ## This is the frequencies
    freqs = tf[band]['resonances'][0]['freq_eta_scan']
    ## Resonant frequency
    fr = tf[band]['resonances'][0]['freq']
    tune_dict = {}
    tune_dict['fname'] = tune_file
    ## Scale the response by the db_offset --> ASK!!
    tune_dict['r'] = r * db_offset
    tune_dict['freqs'] = freqs
    tune_dict['fr'] = fr
    return tune_dict

def fitTuneFile(tune_dict, plot=True):
    if tune_dict['fr'] < max(tune_dict['freqs']) and tune_dict['fr'] > min(tune_dict['freqs']):
        i_fr_0 = np.argmin(abs(tune_dict['freqs']-tune_dict['fr']))
        i_start = min([0, i_fr_0-1500])
        i_end  = max([i_fr_0+1500, len(tune_dict['freqs'])])

    fn = tune_dict['fname']+'_fit'
    
    fine_pars, fine_errs = finefit(tune_dict['freqs'][i_start:i_end], tune_dict['r'][i_start:i_end], tune_dict['fr'], fn, show_plots=plot)
    # print(f'fine_pars: {fine_pars}')
    #print(f'fine_errs: {fine_errs}')
    # print('Qi:', fine_pars['Qr']*( 1-fine_pars['Qr'] / fine_pars['Qc'] ) )
    return fine_pars, fine_errs


def plotFit(tune_dict, fit_dict):

    ##This is the fit parameters plugged int othe fit function.
    iq_fit = resfunc3(tune_dict['freqs'], fit_dict['f0'], fit_dict['Qr'], fit_dict['QcHat'], fit_dict['zOff'], fit_dict['phi'],fit_dict['tau'])

    ##This is also the fit parameters plugged into the fit function, but the x data is only the resonant frequency (so returns one point).
    iq_res = resfunc3(fit_dict['f0'], fit_dict['f0'], fit_dict['Qr'], fit_dict['QcHat'], fit_dict['zOff'], fit_dict['phi'],fit_dict['tau'])

    ## This is the data rotated to the ideal basis
    VNA_ideal_basis = rotate2IdealCompact(tune_dict, fit_dict)
    ## This is the fit rotated to the ideal basis.
    fit_ideal_basis = rotate2IdealSemiCompact(iq_fit, tune_dict['freqs'], fit_dict)
    ## This is the resonance from the fit rotated to the ideal basis.
    iq_res_rotated = rotate2IdealSemiCompact(iq_res,  fit_dict['f0'], fit_dict)

    ## This is the frequency and dissipation basis values, from both the fit and the data.
    VNA_freq, VNA_diss = resonatorBasisEasy(VNA_ideal_basis, fit_dict, axis=None)
    fit_freq, fit_diss = resonatorBasisEasy(fit_ideal_basis, fit_dict, axis=None)
    iq_res_freq, iq_res_diss = resonatorBasisEasy(iq_res_rotated, fit_dict, axis=None)

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
    
    return fig, axs
""" 
def rotate2IdealSemiCompact(z, f, fopt):  
    # osmond
    fr = fopt['f0']
    a =  fopt['zOff']
    tau =  fopt['tau']
    phi = fopt['phi']
    return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))


def rotate2IdealCompact(tune_dict, fopt):  
    # osmond
    z = tune_dict['r']
    f = tune_dict['freqs']
    fr = fopt['f0']
    a =  fopt['zOff']
    tau =  fopt['tau']
    phi = fopt['phi']
    return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))


def resonatorBasisEasy(iq_ideal_basis, fine_pars, axis=-1):
    dS21 = iq_ideal_basis - np.mean(iq_ideal_basis,dtype=complex, axis=axis, keepdims=True)
    frequency =  dS21.imag * fine_pars['Qc'] / ( 2 * fine_pars['Qr'] **2)
    dissipation = dS21.real * fine_pars['Qc']  / (fine_pars['Qr'] **2)
    return frequency, dissipation


######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
################### functions for loading  & processing debug data ###################################
######################################################################################################



######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
############################## functions resonator basis  ############################################
######################################################################################################
"""
def find_closest(vector,value):
    return np.argmin(abs(vector-value))


def plot_noise_and_vna(noise,VNA_z,fit_z=None,f_idx=None,char_zs=None,alpha=0.1,title='',fig_obj=None,save_directory=None,verbose=True):
    if fig_obj == None:
        fig = plt.figure('noise and vna ' + title,figsize=(10,10),dpi=300)
    else:
        fig = fig_obj
    fig.gca();

    plt.title(title + ' complex $S_{21}$')

    ## Plot the VNA result in complex S21
    plt.plot(VNA_z.real,VNA_z.imag,'k',ls='',marker='.',label='VNA',markersize=1)

    ## Plot the noise points
    plt.plot(noise.real,noise.imag,alpha=alpha,marker='.',ls='',label='noise timestream')
    

    if type(char_zs) != type(None):
        plt.plot(char_zs.real,char_zs.imag,\
                 marker='.',ls='--',markersize=10,color='y',label='calibration timestreams')
    if type(fit_z) != type(None):
        plt.plot(fit_z.real,fit_z.imag,ls='-',color='r',label='resonance fit')
    
    if f_idx:
        plt.plot(VNA_z[f_idx].real,VNA_z[f_idx].imag,\
                 ls = '',marker='.',color='r',markersize=10,label='VNA closest to readout frequency')

    real_mean = np.mean(noise.real,axis=0,keepdims=True)
    imag_mean = np.mean(noise.imag,axis=0,keepdims=True)

    if title[-5:] != 'ideal':
        radius_mean = np.sqrt(real_mean**2 + imag_mean**2)

        angles = np.angle(real_mean + 1j*imag_mean)

        dtheta = 0.2

        for idx in range(len(angles)):
            theta_plot = np.linspace(angles[idx] - dtheta,angles[idx] + dtheta, 100)
            x_plot = radius_mean[idx]*np.cos(theta_plot)
            y_plot = radius_mean[idx]*np.sin(theta_plot)
            plt.plot(x_plot,y_plot,color='k',alpha = 0.6,label='arc length direction')

            radius_plot = np.linspace(0.95*radius_mean[idx],1.05*radius_mean[idx],100)
            x_plot = radius_plot*np.cos(angles[idx])
            y_plot = radius_plot*np.sin(angles[idx])
            plt.plot(x_plot,y_plot,ls='-.',color='k',alpha=0.6,label='radius direction')

    plt.plot(real_mean,imag_mean,'g',markersize=10,marker='*',ls='',label='timestream average')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('ADC units')
    plt.ylabel('ADC units')
    plt.axvline(x=0, color='gray')
    plt.axhline(y=0, color='gray')
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    if type(save_directory) != type(None):
        plt.tight_layout()
        plt.savefig(save_directory + title + ' noise and vna.png')

    return fig


def resonator_basis(fine_pars, noise_timestream,readout_f,VNA_f,VNA_z,char_f,char_z,  plot_title=None,verbose=True):

    def rotate_to_ideal(z,f,fr,a,tau,phi):
        return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))

    char_res_idx = int(len(char_f)/2)
    # Define region around this resonance, and fit for parameters
    #index_range = 4000
    #readout_index = find_closest(VNA_f,readout_f)
    
    if readout_f < max(VNA_f) and readout_f > min(VNA_f):
        i_fr_0 = np.argmin(abs(VNA_f - readout_f))
        i_start = min([0, i_fr_0-1500])
        i_end  = max([i_fr_0+1500, len(VNA_f)])
        
    MKID_f = VNA_f[i_start:i_end]
    MKID_z = VNA_z[i_start:i_end]
    
    #fn = fname+'_fit'

    #fine_pars, fine_errs = finefit(MKID_f, MKID_z, readout_f, fn)

    ## Unpack the dictionaries from fine fit result
    fit_fr = fine_pars["f0"]
    fit_Qr = fine_pars["Qr"]
    fit_Qc_hat = fine_pars["QcHat"]
    fit_a   = fine_pars["zOff"]
    fit_phi = fine_pars["phi"]
    fit_tau = fine_pars["tau"]
    fit_Qc  = fine_pars["Qc"]

    fit_Qi = fit_Qr*fit_Qc/(fit_Qc-fit_Qr)




    # Transform VNA to ideal space
    print('fit_a=', fit_a)
    MKID_z_ideal = rotate_to_ideal(MKID_z,MKID_f,fit_fr,fit_a,fit_tau,fit_phi)

    # Transform the VNA fit to ideal space
    fit_f = np.linspace(MKID_f[0],MKID_f[-1],10000)
    fit_z = resfunc3(fit_f, fit_fr, fit_Qr, fit_Qc_hat, fit_a, fit_phi, fit_tau)
    fit_z_ideal = rotate_to_ideal(fit_z,fit_f,fit_fr,fit_a,fit_tau,fit_phi)

    # find some important indices in the f vector
    first_char_fit_idx = find_closest(fit_f,char_f[0])
    last_char_fit_idx = find_closest(fit_f,char_f[-1])

    res_f_idx = find_closest(MKID_f,readout_f)
    res_fit_idx = find_closest(fit_f,readout_f)

    # Find VNA-data subset that covers the characterization-data region in frequency space
    char_region_f = fit_f[first_char_fit_idx:last_char_fit_idx]
    char_region_z = fit_z[first_char_fit_idx:last_char_fit_idx]
    char_region_z_ideal = fit_z_ideal[first_char_fit_idx:last_char_fit_idx]

    # Get the angle (in complex space) of the characterization data
    real_fit = np.polyfit(char_f,np.real(char_z),1)
    imag_fit = np.polyfit(char_f, np.imag(char_z),1)
    char_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))

    # Get the angle (in complex space) of the VNA data
    real_fit = np.polyfit(char_region_f,char_region_z.real,1)
    imag_fit = np.polyfit(char_region_f,char_region_z.imag,1)
    char_region_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))

    # Get the angle (in complex ideal space) of the VNA data
    real_fit = np.polyfit(char_region_f,char_region_z_ideal.real,1)
    imag_fit = np.polyfit(char_region_f,char_region_z_ideal.imag,1)
    char_region_ideal_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))
    
    # Rotate characterization data to VNA data
    char_z_rotated = (char_z-char_z[char_res_idx])\
                     *np.exp(1j*(-1*char_angle+char_region_angle))\
                     +fit_z[res_fit_idx]

    # print(char_z_rotated,char_f,fit_fr,fit_a,fit_tau,fit_phi)
    char_z_rotated_ideal = rotate_to_ideal(char_z_rotated,char_f,fit_fr,fit_a,fit_tau,fit_phi)
    print('fit_a=', fit_a)
    timestream_rotated = (noise_timestream - np.mean(noise_timestream,dtype=complex))\
                      *np.exp(1j*(-1*char_angle+char_region_angle))\
                      +fit_z[res_fit_idx]

    timestream_rotated_ideal = rotate_to_ideal(timestream_rotated,readout_f,fit_fr,fit_a,fit_tau,fit_phi)
    dS21 = timestream_rotated_ideal-np.mean(timestream_rotated_ideal,dtype=complex) #TODO: subtract unbiased mean 

    # Extra rotation for imaginary direction --> frequency tangent direction
    # This is necessary if data_freqs[0] != primary_fr
    # Note that char_f[res_idx] == data_freqs[0] should always be true
    # data_freqs[0] != primary_fr means our vna fit before data taking gave a different fr than our vna fit now

    phase_adjust = 1
    #phase_adjust = np.exp(1j*(np.sign(char_region_ideal_angle)*0.5*np.pi-char_region_ideal_angle))

    frequency = phase_adjust*dS21.imag*fit_Qc/(2*fit_Qr**2)
    dissipation = phase_adjust*dS21.real*fit_Qc/(fit_Qr**2)
    # print(type(frequency[2]),type(dissipation[2]),type(fit_Qc))

    ideal = {}
    ideal['f'] = MKID_f
    ideal['z'] = MKID_z_ideal
    ideal['char z'] = char_z_rotated_ideal
    ideal['timestream'] = timestream_rotated_ideal
    ideal['fit f'] = fit_f
    ideal['fit z'] = fit_z_ideal

    resonator = {}
    resonator['fr'] = fit_fr
    resonator['Qr'] = fit_Qr
    resonator['Qc'] = fit_Qc
    resonator['a'] = fit_a
    resonator['phi'] = fit_phi

    if plot_title is not None:
        print('plotting...')
        fig = plot_noise_and_vna(timestream_rotated_ideal,MKID_z_ideal,
                               fit_z=fit_z_ideal, f_idx=res_f_idx,alpha=.05,
                               char_zs=char_z_rotated_ideal,title='ideal space',verbose=False)
        
    else: fig = plt.figure()
    return frequency, dissipation, ideal, resonator, fig

"""
######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
############################## functions for aligning pulses  ########################################
######################################################################################################


def fitMins( chunked_alignment_data, num_pulse_types, pulses_to_fit_shift):
    if len(pulses_to_fit_shift) == 0: 
        pulses_to_fit_shift = np.arange(num_pulse_types)
    choices  = pulse_choices(num_pulse_types, len(chunked_alignment_data), pulses_to_fit_shift)
    chunked_alignment_data_for_fit = chunked_alignment_data[choices] #only has chunks for pulses you want to use to determine your shift
    x_array = np.arange(len(chunked_alignment_data))[choices] #nchunks, but only ones corresponding to pulses we are using
    
    min_idx_data = []  # will be populated with idx of minimum vaule of each chunk
    for i in range(len(chunked_alignment_data_for_fit)): 
        index = np.argmin(chunked_alignment_data_for_fit[i,:])
        min_idx_data.append(index)
    min_idx_data = np.array(min_idx_data) #if not using all pulses, this is SHORTER than the total num_chunks
    #print(f'total chunks: {len(chunked_alignment_data)},number of chunks used in fit:{len(min_idx_data)}') 
    p_fit= np.polyfit(x_array, min_idx_data, 1) #these are the params for the fit
    #print("p_fit:", p_fit)
    fit = np.poly1d(p_fit) #this is a function, seeded with the output of the fit
    ## fit takes input of rows, and gives output of where min is located 
    return fit

def fitMaxs(chunked_alignment_data, num_pulse_types, pulses_to_fit_shift):
    if len(pulses_to_fit_shift) == 0: 
        pulses_to_fit_shift = np.arange(num_pulse_types)
    choices  = pulse_choices(num_pulse_types, len(chunked_alignment_data), pulses_to_fit_shift)
    
    chunked_alignment_data_for_fit = chunked_alignment_data[choices] #only has chunks for pulses you want to use to determine your shift
    x_array = np.arange(len(chunked_alignment_data))[choices] #nchunks, but only ones corresponding to pulses we are using
    
    max_idx_data = []  # will be populated with idx of minimum vaule of each chunk
    for i in range(len(chunked_alignment_data_for_fit)): 
        index = np.argmax(chunked_alignment_data_for_fit[i,:])
        max_idx_data.append(index)
    max_idx_data = np.array(max_idx_data) #if not using all pulses, this is SHORTER than the total num_chunks
    #print(f'total chunks: {len(chunked_alignment_data)},number of chunks used in fit:{len(min_idx_data)}') 
    p_fit= np.polyfit(x_array, max_idx_data, 1) #these are the params for the fit
    #print("p_fit:", p_fit)
    fit = np.poly1d(p_fit) #this is a function, seeded with the output of the fit
    ## fit takes input of rows, and gives output of where max is located 
    return fit

def pulse_choices(num_pulse_types, num_chunks, pulse_choice):
    """
    num_pulse_types: (int) number of pulses sent in (used for sawtooth x square wave) assumes pulse amplitude repeats at this rate
    num_chunks: (int) number of chunks you have split your data into 
    pulse_choice: (int) which pulse to look at... (used for sawtooth x square wave) must be < num_pulse_types 

    returns: choices, an array of indices that correspond to the pulses of choice.
    """
    choice_array_fit = np.arange(num_chunks*num_pulse_types).reshape(num_chunks,num_pulse_types).T  
    mask_fit = np.ma.masked_greater_equal(choice_array_fit, num_chunks)
    choices = mask_fit[pulse_choice].compressed()   
    return choices


######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
############################## functions shotnoise calculation  ######################################
######################################################################################################

def gaus(X,C,X_mean,sigma):
    return C*np.exp(-(X-X_mean)**2/(2*sigma**2))


def Pulse2(t, tau1, tau2, offset, t0=0):
    pulse = np.zeros(len(t))
    dt = t - t0
    m = (dt>0)
    pulse[m] = (np.exp(-dt[m]/tau1)-np.exp(-dt[m]/tau2))
    norm = 1/((tau2/tau1)**(tau2/(tau1-tau2)) - (tau2/tau1)**(tau1/(tau1-tau2)))
    pulse *= norm
    pulse += offset
     
    return pulse

def decimate(data, num_decimate):
    total=len(data)
    tofill=np.zeros(int(total/num_decimate))
    i=0
    while (i < (int(total/num_decimate))):
        tofill[i]=np.mean(data[i*num_decimate:(i+1)*num_decimate])
        i+=1
    return tofill

