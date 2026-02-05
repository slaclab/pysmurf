# now the Imports: 
import sys
import os
import glob 
import gc 

import json
import h5py

import numpy as np
from scipy import signal  #import butter, welch, filtfilt, periodogram, savgol_filter
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pylab as plt

import datetime
import time
sys.path.append('/usr/local/src/pysmurf/scratch/smithzj/rc3_code/')
from utils.ResonanceFitter import *
from utils.data_collection_utils import rotate2IdealSemiCompact, rotate2IdealCompact, fitMins, fitMaxs, pulse_choices, plot_fit


class debugDatAnalysis:
    def __init__(self, S, filename, directory, file_dict, pre_trigger_s,Eabs_conversion_factor, given_fit_dict=None,  PSD_lo_f=1e2, PSD_hi_f=5e4,rchannel=None, db_offset=1, show_fit_plots=True, threshold=2e-7): 
        """
        A class to handle the loading, processing, and analysis of debug data from SMuRF (Superconducting Microresonator Firmware).

        This class integrates various methods to decimate, fit, align, and process resonator data from various sources,
        including pysmurf tune files (VNA sweeps), characterization tone averages, single .dat files taken in pysmurfs takeDebugData mode.
        The final data can be analyzed in multiple bases, including the resonator and quasiparticle basis.
        NOTE: the char tone averages MUST include a resonant characterization tone. 
        Attributes:
        -----------
        filename : str
            The name of the debug data file.
        dirname : str
            The directory path where the debug data is stored.
        file_dict : dict
            A dictionary containing information about the data such as frequency, sampling rate, etc.
        band : int
            The frequency band of the resonator.
        rchannel : int
            The readout channel of the resonator data.
        orchannels : list of int
            Off-resonance channels for data cleaning.
        db_offset : int
            The decibel offset applied to the data.
        PSD_lo_f : float
            The lower bound frequency for Power Spectral Density (PSD) analysis.
        PSD_hi_f : float
            The upper bound frequency for PSD analysis.
        data_temp : float
            The temperature of the data acquisition environment.
        MB_results : list
            Results from Mattis-Bardeen fitting.
        pre_trigger_s : float
            The time before the trigger used for alignment in seconds.

        Methods:
        --------
        __init__:
            Initializes the class and sets the necessary attributes.

        loadDebugDataAndConvert:
            Loads SMuRF debug data and converts it to complex IQ format.

        decimateData:
            Decimates and averages the data to reduce its resolution and sampling rate.

        fitData:
            Loads and fits a tune file to obtain the resonator parameters.

        loadCharTones:
            Loads characterization tones from the file dictionary.

        plotCharTones:
            Plots characterization tones on top of fitted resonator data.

        alignChunks:
            Aligns chunks of data for processing pulses based on peak fitting.

        plot_chunks:
            Plots the chunked data with vertical lines marking the pre-trigger points.

        chunkyData:
            Converts flat streaming data into evenly sized chunks.

        averageDecimate:
            Decimates the data using an averaging method.

        chunkDataByNumChunks:
            Splits the data into a 3D array with a specified number of chunks.

        
        """
        self.filename = filename
        self.dirname = directory
        self.file_dict = file_dict
        self.band = file_dict['band']
        self.rchannel = rchannel
        self.db_offset = db_offset       
        self.PSD_lo_f = PSD_lo_f
        self.PSD_hi_f = PSD_hi_f
        self.pre_trigger_s = pre_trigger_s
        self.num_pulses =  len(self.file_dict['awg_settings']['amplitudes'])
        self.full_filename = self.dirname +'/' + self.filename 
        print(f'PROCESSING {self.full_filename}')

        #PROCESS THE TUNE FILE, IF NECESSARY
        # 0. Load tune file, and make a fit
        print("LOADING TUNE FILE...")
        ### TODO: MAKE THIS LOAD FIT DICT IF IT ALREADY EXISTS

        try:
            self.tune_dict = self.loadTuneFile( tunefile=self.file_dict['tunefile'], band=self.band, db_offset=self.db_offset) 
            self.tunefile = self.file_dict['tunefile']
        except FileNotFoundError:
            larger_dir = '/'.join(self.dirname.split('/')[:-3])
            tune_file = larger_dir +'/tune/'+ self.file_dict['tunefile'].split('/')[-1]
            self.tune_dict = self.loadTuneFile( tunefile=tune_file, band=self.band, db_offset=self.db_offset)
            self.tunefile = tune_file
        if isinstance(given_fit_dict, type(None)):
            print("FITTING TUNE FILE")
            self.fitData(plot=show_fit_plots)
        else: 
            print("USING GIVEN FIT DICT") 
            self.fit_dict = given_fit_dict
            

        # LOAD YOUR CHAR TONES
        self.loadCharTones() # will add charzs and charfs as attributes
        if show_fit_plots:
            self.plotCharTones()
        # LOAD THE DATA 
        self.rIQ = self.loadDebugDataAndConvert(self.full_filename, S, self.rchannel) 
        self.fs  = self.file_dict['fs'] #Hz 
        
        # GET A COARSE IDEAL BASIS 
        ideal, resonator, fig= self.ideal_basis(self.fit_dict, \
                                                self.rIQ,\
                                                self.file_dict['freq_in_Hz']*1e-6,\
                                                self.tune_dict['freqs'],\
                                                self.tune_dict['r'],\
                                                np.array(self.char_f),\
                                                np.array(self.char_z),\
                                                avg_start_idx=0,\
                                                avg_stop_idx=None,\
                                                plot_title=None,verbose=True)

        #ideal_Q = ideal['timestream'].imag
        coarse_dff, coarse_dis = self.resonatorBasisFromIdeal(ideal['timestream'], resonator, avg_start_idx=0,avg_stop_idx=None )
        
        # GET AN "AVERAGE PULSE" OF EACH TYPE
        coarse_chunked_dff = self.chunkyData(coarse_dff, fs_Hz=self.fs, start=0)
        average_chunks_dff = []
        
        maxs = []
        for n in range(self.num_pulses):
            mean = np.mean(coarse_chunked_dff[n::self.num_pulses], axis=0)
            average_chunks_dff.append(mean)
            maxs.append(np.max(mean))
            if show_fit_plots: 
                plt.plot(mean, label=n)
                plt.hlines(np.max(mean), 0, len(mean))

        if show_fit_plots:
            plt.title(f'Average pulse of given type')
            plt.legend()
            plt.show()
            
        # FIGURE OUT WHICH PULSE IS THE "BIGGEST" in order to align
        biggest_chunk = np.argmax(np.array(maxs))
        if show_fit_plots: 
            for n in range(self.num_pulses):
                plt.plot(coarse_chunked_dff[biggest_chunk+n], label=n)
            plt.title(f'individual pulses of a given type, biggest_chunk={biggest_chunk}')
            plt.legend()
            plt.show()
        time_chunk = np.arange(len(average_chunks_dff[0]))/(self.fs)
        pulse2align2 = average_chunks_dff[biggest_chunk]
        if show_fit_plots:
            plt.plot(time_chunk, pulse2align2)
            plt.title('pulse2align2')
            plt.show()
            plt.plot(time_chunk, average_chunks_dff[(biggest_chunk+1)%self.num_pulses])
            plt.title('next pulse')
        #plt.show()
        # FIND THE RISING EDGE, TO DETERMINE IF THE
        i1 =  self.find_rising_edge_index(pulse2align2, threshold=threshold)
        ## NOW SHIFT AND CHUNK THE DATA, AND SEE HOW WELL YOU HAVE DONE
        idx2shift = 0 
        if biggest_chunk == 0: 
            idx2shift += len(pulse2align2) * self.num_pulses 
        else:  
            idx2shift += len(pulse2align2) * biggest_chunk 
        idx2shift += i1 
        pre_trigger_idx =  int(self.pre_trigger_s  *  self.fs ) #s * fs_Hz = samples 
        idx2shift -=   pre_trigger_idx 
        idx2shift = int(idx2shift)
        shifted_rIQ = self.rIQ[idx2shift:]
        shifted_chunked_rIQ = self.chunkyData(shifted_rIQ, fs_Hz=self.fs, start=0)
        if show_fit_plots:
            plt.vlines(time_chunk[i1], min(average_chunks_dff[(biggest_chunk+1)%self.num_pulses]),max(average_chunks_dff[(biggest_chunk+1)%self.num_pulses]))
            plt.show()
            plt.plot(time_chunk, shifted_chunked_rIQ[0])
            plt.plot(time_chunk, np.mean(shifted_chunked_rIQ[0::self.num_pulses], axis=0))
            plt.title('shifted, chunked rIQ')
            plt.show()
        self.rIQ_sorted = self.splitIntoPulses(shifted_chunked_rIQ)
        ideal, resonator, fig = self.ideal_basis(self.fit_dict, \
                                                 shifted_chunked_rIQ,\
                                                 self.file_dict['freq_in_Hz']*1e-6,\
                                                 self.tune_dict['freqs'],\
                                                 self.tune_dict['r'],\
                                                 np.array(self.char_f),\
                                                 np.array(self.char_z),\
                                                 avg_start_idx=0,\
                                                 avg_stop_idx=int(pre_trigger_idx*0.8),\
                                                 plot_title=None,verbose=True)
        
        self.ideal_I = ideal['timestream'].real
        self.ideal_Q = ideal['timestream'].imag

        ### 10. Now calculate the resontator basis data from the 2D ideal IQ data we just found.
        print("CALCULATING RESONATOR BASIS FROM IDEAL")
        self.dff, self.dis = self.resonatorBasisFromIdeal(self.ideal_I +1j*self.ideal_Q, resonator, avg_start_idx=0,avg_stop_idx=int(pre_trigger_idx*0.8) )

        #dff_dec, time_s_dec, fs_dec= self.decimateData(data=self.dff.flatten(), update_attributes=False, return_values=True) # adds attributes time_s, and update fs,
        #dis_dec, time_s_dec, fs_dec= self.decimateData(data=self.dis.flatten(), update_attributes=True, return_values=True) # adds attributes time_s, and update fs,
        
        #dff_dec = self.chunkyData(dff_dec, fs_Hz=fs_dec, start=0)
        #dff_dec = self.chunkyData(dis_dec, fs_Hz=fs_dec, start=0)
        
        ### 11. finally, sort all of the basis data into useful pulse dictionaries 
        self.dff_sorted  = self.splitIntoPulses(self.dff)
        self.dis_sorted  = self.splitIntoPulses(self.dis)
        #self.ideal_I_sorted = self.splitIntoPulses(self.ideal_I)
        #self.ideal_Q_sorted = self.splitIntoPulses(self.ideal_Q)
          
        # 12. (bonus) rotate into the quasiparticle basis
        self.Eabs_sorted = self.splitIntoPulses(Eabs_conversion_factor * self.dff)

        #del self.rIQ
        #del average_chunks_Q
        #del coarse_chunked_Q
        #del self.ideal_I
        #del self.ideal_Q
        #del self.dff
        #del self.dis
        del ideal
        del resonator
        del coarse_dff
        del coarse_dis
        #del self.rIQ_sorted
        gc.collect()
        # this class is now "done..." can modify Elizabeth's class to process whole sweeps. 
        ####################################################################################
        ####################################################################################
    def plotFit(self, tune_dict, fit_dict):
    
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

    def find_rising_edge_index(self, pulse, threshold=1e-7):
        """
        Find the index where the signal crosses a rising edge threshold.
        threshold_fraction is the fraction of the pulse's max amplitude.
        """
        #max_val = np.max(pulse)
        #threshold = threshold_fraction * max_val
        for i in range(1, len(pulse)):
            if pulse[i-1] < threshold <= pulse[i]:
                return i
        return None 
    def find_steepest_rising_edge_index(self, pulse, window_length=21, polyorder=4):
        """
        Use Savitzky-Golay filter to compute smoothed derivative for robust edge detection.
        """
        # window_length must be odd and >= polyorder + 2
        deriv = signal.savgol_filter(pulse, window_length, polyorder, deriv=1)
        max_slope_index = np.argmax(deriv)
        return max_slope_index
        
    def removeRowFromSortedAttr(self, sorted_attribute, pulse_type, row):
        if pulse_type in sorted_attribute.keys():
            if row < len(sorted_attribute[pulse_type]):
                print(f'deleting {row}')
                sorted_attribute[pulse_type] = np.delete(sorted_attribute[pulse_type], (row), axis=0)
        return sorted_attribute
    def removeRowFromAllSortedAttr(self,  pulse_type, row):
        if hasattr(self, 'dff_sorted'):
            self.dff_sorted = self.removeRowFromSortedAttr(self.dff_sorted, pulse_type, row)
        if hasattr(self, 'dis_sorted'):
            self.dis_sorted = self.removeRowFromSortedAttr(self.dis_sorted, pulse_type, row)
        if hasattr(self, 'dnqp_k1_sorted'):
            self.dnqp_k1_sorted = self.removeRowFromSortedAttr(self.dnqp_k1_sorted, pulse_type, row)
        if hasattr(self, 'dnqp_k2_sorted'):
            self.dnqp_k2_sorted  = self.removeRowFromSortedAttr(self.dnqp_k2_sorted, pulse_type, row)
                    
    def grabPulse(self, basis, pulse_type):
        """
        basis: pick betwen "dff", "dis", "dnqp_k1",  "dnqp_k2"
        pulse_type: integer less than total number of pulses
        """
        ### TODO: check if pulse type exists, return zero array and alert if not
        if basis == 'dff':
            return self.dff_sorted[pulse_type]
        elif basis == 'dis':
            return self.dis_sorted[pulse_type]
        elif basis == 'dnqp_k1':
            return self.dnqp_k1_sorted[pulse_type]
        elif basis == 'dnqp_k2':
            return self.dnqp_k2_sorted[pulse_type]
        elif basis == 'ideal_I':
            return self.ideal_I_sorted[pulse_type]
        elif basis == 'ideal_Q':
            return self.ideal_Q_sorted[pulse_type]

    ### methods for loading debug data and decimating: 
    def loadDebugDataAndConvert(self, debug_data_file, S, subband_half_width_mhz, channel=None):
        """
            channel: None -> single channel mode, Int -> multi channel mode, what channel to load (0-512)
            loads debug data and makes known necessary smurf conversion to agree with eta scan
            returns complex IQ
         """
        i, q, sync =  S.decode_single_channel(debug_data_file,  swapFdF= False) # Smurf function for multi tone debug data: 
        n_subbands = S.get_number_sub_bands()
        digitizer_frequency_mhz = S.get_digitizer_frequency_mhz()
        subband_half_width_mhz = (digitizer_frequency_mhz / n_subbands)
        if channel:
            print('picking chan:', channel)
            i, q, sync = S.decode_data(debug_data_file)
        I = i / (subband_half_width_mhz)
        Q = q / (-1*subband_half_width_mhz)
        return I + 1j*Q  
    def decimateData(self, data, time_s=None, decimation=None,update_attributes=True, return_values=False):
        """
        Decimate the data by reducing its resolution.

        Parameters:
        -----------
        data : ndarray
            Input data to decimate.
        time_s : ndarray, optional
            Time values associated with the data (default: None).
        decimation : int, optional
            Decimation factor, if not provided it is calculated from sampling rate and PSD_hi_f.
        update_attributes : bool, optional
            If True, updates the class attributes like time_s, and fs (default: True).
        return_values : bool, optional
            If True, returns the decimated data and time values (default: False).

        Returns:
        --------
        data_dec : ndarray
            Decimated data.
        time_s_dec : ndarray
            Decimated time values.
        fs_dec : int
            New decimated sampling rate.
        """
        if decimation == None:
            decimation =  int(self.file_dict['fs']//self.PSD_hi_f)
        try:
            if not time_s: 
                time_s = np.arange(len(data.real)) * (1/(self.file_dict['fs']))
        except: print("time_s given for decimation")
            
        ## need to chop data to be evenly decimated
        decimation = int(decimation)
        print("decimation by", decimation)
        data = data[0:int(len(data)//decimation * decimation)]
        data_dec = self.averageDecimate(data, decimation)
        data_dec = data_dec.flatten()
        time_s_dec = time_s[::decimation]
        fs_dec = int(self.file_dict['fs'] / decimation) 
        if update_attributes:
            print('updating rIQ, time_s, fs after decimating')
            #self.rIQ = data_dec
            self.time_s = time_s_dec
            self.fs = fs_dec #Hz 
            
        if return_values:
            return data_dec, time_s_dec, fs_dec
    def averageDecimate(self,timestream,decimation):
        '''
        decimation code from alvaro's "noise_removal" function
    
        input: timestream (I think it needs to be real), decimation factor
        output: decimated timestream
        '''
        # ids = np.arange(len(timestream))//decimation
        # timestream_decimated = np.bincount(ids,timestream)/np.bincount(ids)
        dtype = timestream.dtype
        decimation_len = int(len(timestream)/decimation)
        #print(f"decimation_len={np.shape(decimation_len)}")
        chunked_timestream = self.chunkDataByNumChunks(timestream,decimation_len)
        timestream_decimated = np.mean(chunked_timestream,axis=0,dtype=dtype)
        #print(f"np.shape(timestream_decimated)={np.shape(timestream_decimated)}")
        return timestream_decimated
    def chunkDataByNumChunks(self, timestreams,num_chunks):
        """
        will take a LxN array of L-length timestreams of N frequencies and recast it
        into a 3D array that is (L/num_chunks) x num_chunks x N
    
        input: timestream is a numpy array, num_chunks is an integer
        output: a 3D array
        """
        dtype = timestreams.dtype
        L = int(timestreams.shape[0])
        if timestreams.ndim == 1:
            timestreams = np.expand_dims(timestreams,axis=1)
        N = int(timestreams.shape[1])
        if L%num_chunks != 0:
            raise Exception('timestream must be divisible into equal sized chunks')
        chunk_L = int(L / num_chunks)
        chunked_timestreams = np.zeros((chunk_L,num_chunks,N),dtype=dtype)
        for freq_idx in range(N):
            timestream = timestreams[:,freq_idx]
            chunked_timestream = np.reshape(timestream,(chunk_L,num_chunks),order='F')
    
            chunked_timestreams[:,:,freq_idx] = chunked_timestream
        return chunked_timestreams       
    
    ### methods for processing tune files
    def loadTuneFile(self, tunefile=None, band=None, db_offset=None):  
        if not tunefile: 
            tunefile = self.tunefile
        if not band: 
            band = self.band 
        if not db_offset:
            db_offset = self.db_offset 
        """
        Parameters: 
        -----------
        tunefile: (str) the full file path to tune file
        band: (int) what band was data taken on? 
        

        Returns:
        -------- 
        returns dict of file name, iq scan, freqs, and res freq 
        """
        tf =  np.load(tunefile, allow_pickle=True).item()
        ## This is the response - so the transmission for I and Q
        r = tf[band]['resonances'][0]['resp_eta_scan']
        ## This is the frequencies
        freqs = tf[band]['resonances'][0]['freq_eta_scan']
        ## Resonant frequency
        fr = tf[band]['resonances'][0]['freq']
        tune_dict = {}
        tune_dict['fname'] = tunefile
        ## Scale the response by the db_offset --> ASK!!
        tune_dict['r'] = r * db_offset
        tune_dict['freqs'] = freqs
        tune_dict['fr'] = fr
        return tune_dict
      
    ### methods for processing char tones:
    def loadCharTones(self, file_dict=None):
        try: 
            if not file_dict:
                file_dict = self.file_dict
        except: print('file_dict given')
        charzs = []
        charfs = []
        for f in file_dict['char_avgs'].keys():
            chavg = file_dict['char_avgs'][f]
            charzs.append(chavg[0]/1.2 + 1j*chavg[1]/(-1.2))
            charfs.append(float(f))
        self.char_z = charzs
        self.char_f = charfs
        return       
    def plotCharTones(self, tune_dict=None, fit_dict=None, char_f=None, char_z=None):
        try: 
            if not tune_dict:
                tune_dict = self.tune_dict
        except: print('tune_dict given')
        try: 
            if not fit_dict:
                fit_dict = self.fit_dict
        except: print('fit_dict given')
        try: 
            if not char_f:
                char_f = self.char_f
        except: print('char_f given')
        try: 
            if not char_z:
                char_z = self.char_z
        except: print('char_z given')

        fig, axs = self.plotFit(tune_dict, fit_dict)
        for i in range(len(char_f)):
            charz = char_z[i]
            axs[0].plot(charz.real, charz.imag, marker='o', color='k')
            axs[0].plot(charz.real, charz.imag, marker='*')
        return fig, axs
    
    ### methods for chunking and aligning data
    def chunkyData(self, data, fs_Hz=None, chunk_duration_s=None, start=0):
        """
        fs_MHz: sample freq, MHz
        chunk_duration_us: period of wave on LED 
        data: np array with shape: (n_samp,) data to chunk up 
        start: idx to start your chunks on, so you can center pulses in the window 
        returns: raw data, separated into chunks. 
        """
        if not fs_Hz:
            fs_Hz = self.fs 
        if not chunk_duration_s:
            if 'chunk_duration_s' in self.file_dict['awg_settings'].keys():
                chunk_duration_s = self.file_dict['awg_settings']['chunk_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
            else:
                chunk_duration_s = self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
        n_samp = len(data)
        # Calculate the number of samples per chunk
        samples_per_chunk = int(fs_Hz * chunk_duration_s)
        num_chunks = n_samp // samples_per_chunk 
        #remove the end of the data, so you can have divisible chunks
        data_chunks = data[start:num_chunks*samples_per_chunk+start]
    
        # Reshape the array into chunks
        data_chunks = data_chunks.reshape(num_chunks, (samples_per_chunk))
        
        return data_chunks 

    ### methods for resonator basis: 
    def find_closest(self, vector,value):
        return np.argmin(abs(vector-value))
        
    def ideal_basis(self, fine_pars, noise_timestream,readout_f,VNA_f,VNA_z,char_f,char_z, avg_start_idx=0, avg_stop_idx=None, plot_title=None,verbose=True):
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
        MKID_z_ideal = rotate_to_ideal(MKID_z,MKID_f,fit_fr,fit_a,fit_tau,fit_phi)
        # Transform the VNA fit to ideal space
        fit_f = np.linspace(MKID_f[0],MKID_f[-1],10000)
        fit_z = resfunc3(fit_f, fit_fr, fit_Qr, fit_Qc_hat, fit_a, fit_phi, fit_tau)
        fit_z_ideal = rotate_to_ideal(fit_z,fit_f,fit_fr,fit_a,fit_tau,fit_phi)
    
        # find some important indices in the f vector
        first_char_fit_idx = self.find_closest(fit_f,char_f[0])
        last_char_fit_idx = self.find_closest(fit_f,char_f[-1])
    
        res_f_idx = self.find_closest(MKID_f,readout_f)
        res_fit_idx = self.find_closest(fit_f,readout_f)
    
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

        if len(np.shape(noise_timestream))==2:
            mean = np.mean(noise_timestream[:,avg_start_idx:avg_stop_idx], axis=-1, keepdims=True) #axis=-1
        elif len(np.shape(noise_timestream))==1:
            mean = np.mean(noise_timestream[avg_start_idx:avg_stop_idx],  axis=-1, keepdims=True) #axis=-1,

        #print(f'mean: {mean}')
        timestream_rotated = (noise_timestream - mean) * np.exp(1j * (-1 * char_angle + char_region_angle)) + fit_z[res_fit_idx]
        
        timestream_rotated_ideal = rotate_to_ideal(timestream_rotated,readout_f,fit_fr,fit_a,fit_tau,fit_phi)
    
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
            fig = self.plot_noise_and_vna(timestream_rotated_ideal,MKID_z_ideal,
                                   fit_z=fit_z_ideal, f_idx=res_f_idx,alpha=.05,
                                   char_zs=char_z_rotated_ideal,title='ideal space',verbose=False)
            
        else: fig = plt.figure()
        return ideal, resonator, fig

    def resonatorBasisFromIdeal(self, timestream_rotated_ideal, resonator_fit, avg_start_idx=0, avg_stop_idx=None):
        
        if len(np.shape(timestream_rotated_ideal)) == 2:
            mean = np.mean(timestream_rotated_ideal[:,avg_start_idx:avg_stop_idx], axis=-1, keepdims=True) #axis=-1,
        elif len(np.shape(timestream_rotated_ideal)) == 1:
            mean = np.mean(timestream_rotated_ideal[avg_start_idx:avg_stop_idx],   axis=-1, keepdims=True) #axis=-1,
            
        dS21 = timestream_rotated_ideal - mean
        fit_Qc = resonator_fit['Qc']
        fit_Qr = resonator_fit['Qr']
        # Extra rotation for imaginary direction --> frequency tangent direction
        # This is necessary if data_freqs[0] != primary_fr
        # Note that char_f[res_idx] == data_freqs[0] should always be true
        # data_freqs[0] != primary_fr means our vna fit before data taking gave a different fr than our vna fit now
    
        phase_adjust = 1
        #phase_adjust = np.exp(1j*(np.sign(char_region_ideal_angle)*0.5*np.pi-char_region_ideal_angle))
    
        frequency = phase_adjust*dS21.imag*fit_Qc/(2*fit_Qr**2)
        dissipation = phase_adjust*dS21.real*fit_Qc/(  fit_Qr**2)
        return frequency, dissipation

    def plot_noise_and_vna(self, noise,VNA_z,fit_z=None,f_idx=None,char_zs=None,alpha=0.1,title='',fig_obj=None,save_directory=None,verbose=True):
        noise = np.mean(noise, axis=0)
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

        if type(fit_z) != type(None):
            plt.plot(fit_z.real,fit_z.imag,ls='-',color='r',linewidth=1, label='resonance fit')
    
        if type(char_zs) != type(None):
            plt.plot(char_zs.real,char_zs.imag,\
                     marker='.',ls='--',markersize=10,color='y', label='calibration timestreams')
        
        
        if f_idx:
            plt.plot(VNA_z[f_idx].real,VNA_z[f_idx].imag,\
                     ls = '',marker='.',color='r',markersize=5,label='VNA closest to readout frequency')
    
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
    
        plt.plot(real_mean,imag_mean,'c',markersize=3,marker='*',ls='',label='timestream average')
    
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

    def pulseChoices(self, num_pulse_types, num_chunks, pulse_choice):
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
    def splitIntoPulses(self, chunked_array, num_pulses=None):
        if not num_pulses: 
            num_pulses = len(self.file_dict['awg_settings']['amplitudes'])
        masks = {}
        for n in range(num_pulses):
            choices = self.pulseChoices(num_pulses, len(chunked_array), n)
            masks[n] = chunked_array[choices]
        return masks



