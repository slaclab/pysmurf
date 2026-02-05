import numpy as np
import os 
import matplotlib
import matplotlib.pylab as plt
import glob
from scipy.optimize import curve_fit
import re
import h5py
import json


print(os.getcwd())
from utils.ResonanceFitter import *
from utils.data_collection_utils import rotate2IdealSemiCompact, rotate2IdealCompact, fitMins, fitMaxs, pulse_choices
import utils.MB_equations as MBe



class debugDatAnalysis:
    def __init__(self, S, filename, directory, file_dict, pre_trigger_s,  PSD_lo_f=1e2, PSD_hi_f=5e4, MB_results=None, data_temp=None, rchannel=None, orchannels=None, db_offset=1,show_fit_plots=True): 
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

        getBigIdxAlignmentData:
            Finds the index of the largest pulse in the dataset for alignment purposes.

        makeShiftedChunks:
            Shifts and chunks the streaming data to align pulses based on the big pulse.

        shiftChunks:
            Shifts the streaming data by a known number of indices and chunks it.

        quasiparticleBasis:
            Converts data into the quasiparticle basis using Mattis-Bardeen theory.

        plot_chunks:
            Plots the chunked data with vertical lines marking the pre-trigger points.

        chunkyData:
            Converts flat streaming data into evenly sized chunks.

        averageDecimate:
            Decimates the data using an averaging method.

        chunkDataByNumChunks:
            Splits the data into a 3D array with a specified number of chunks.

        getBigIdx:
            Identifies the chunk with the highest amplitude (the 'big' pulse).
        """
        self.filename = filename
        self.dirname = directory
        self.file_dict = file_dict
        self.band = file_dict['band']
        self.rchannel = rchannel
        self.orchannels = orchannels 
        self.db_offset = db_offset       
        self.PSD_lo_f = PSD_lo_f
        self.PSD_hi_f = PSD_hi_f
        self.data_temp = data_temp
        self.MB_results = MB_results
        self.pre_trigger_s = pre_trigger_s
        
        # 2. Load the data and average-decimate
        self.full_filename = self.dirname +'/' + self.filename 
        print(f'PROCESSING {self.full_filename}')
        self.rIQ = self.loadDebugDataAndConvert(self.full_filename, S, self.rchannel) 
        self.fs  = self.file_dict['fs'] #Hz 
        self.decimateData(data=self.rIQ, update_attributes=True) # adds attributes time_s, and update fs and rIQ, 
        
        # 3. clean the data ##TODO
        if self.orchannels is not None: 
            self.orIQs = self.loadDebugDataAndConvert(self.full_filename,S, self.orchannels) #ELIZABETH LOOK HERE FOR 2 TONE
            print("ALERT: gave off resonant tones but dont have data cleaning working... doing nothing ")
        else: print("ONLY ONE TONE GIVEN, NOT CLEANING DATA")

        # 4. Load tune file, and make a fit
        print("LOADING AND FITTING TUNE FILE...")
        self.tunefile = self.file_dict['tunefile']
        self.fitData(plot=show_fit_plots)

        # 5. Overplot char avgs to see how skewed they are
        print("LOADING AND PLOTTING CHARACTERIZATION TONES")
        self.loadCharTones() # will add charzs and charfs as attributes
        #self.plotCharTones()
        
        # 6. make char avgs adjustments, get to the ideal and freq/Q basis.
        print("ROTATING TO IDEAL BASIS")

        ### get data into a slightly biased ideal basis, from 1D data which is what bigIDxAlingmentData needs 
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
        ideal_I = ideal['timestream'].real
        ideal_Q = ideal['timestream'].imag
        
        # 7. use that ideal_I data to do the whole chunk pulse alignment / reordering shenanigans
        # since we KNOW a pulse will look like a peak in the ideal basis. 
        print('FINDING ALIGNMENT DATA FROM A PEAK')
        self.getBigIdxAlignmentData(alignment_data=ideal_Q) # adds attribute bigIdx (what is the first 'chunk' with the biggest pulse)

        # 8. Now, we shift our data to have the Big pulse in the correct location
        shifted_chunked_Q = self.makeShiftedChunks(ideal_Q, peak=True) ## adds attribute shift_idxs_2_cut
        shifted_chunked_rIQ = self.shiftChunks(self.rIQ) 

        # 9. Now, go in and recalculate our ideal IQ data, 
        # using a mean calculated from the pretrigger region of each pulse (each pulse gets its own mean subtracted).
        pre_trigger_idx =  int(self.pre_trigger_s  *  self.fs ) #s * fs_Hz = samples 
        ideal, resonator, fig = self.ideal_basis(self.fit_dict, \
                                                                             shifted_chunked_rIQ,\
                                                                             self.file_dict['freq_in_Hz']*1e-6,\
                                                                             self.tune_dict['freqs'],\
                                                                             self.tune_dict['r'],\
                                                                             np.array(self.char_f),\
                                                                             np.array(self.char_z),\
                                                                             avg_start_idx=0,\
                                                                             avg_stop_idx=int(pre_trigger_idx*0.8),\
                                                                             plot_title=self.filename,verbose=True)
        
        plt.show()
        self.ideal_I = ideal['timestream'].real
        self.ideal_Q = ideal['timestream'].imag
        

        ### 10. Now calculate the resontator basis data from the 2D ideal IQ data we just found.
        print("CALCULATING RESONATOR BASIS FROM IDEAL")
        self.dff, self.dis = self.resonatorBasisFromIdeal(self.ideal_I +1j*self.ideal_Q, resonator, avg_start_idx=0,avg_stop_idx=int(pre_trigger_idx*0.8) )
        #self.aligned_dff = self.align_chunks(streaming_chunks=dff_chunks, alignment_chunks=dff_chunks)
        
        ### 11. finally, sort all of the basis data into useful pulse dictionaries 
        self.dff_sorted  = self.splitIntoPulses(self.dff)
        self.dis_sorted  = self.splitIntoPulses(self.dis)
        self.ideal_I_sorted = self.splitIntoPulses(self.ideal_I)
        self.ideal_Q_sorted = self.splitIntoPulses(self.ideal_Q)

          
        # 12. (bonus) rotate into the quasiparticle basis
        
        if self.MB_results and self.data_temp: 
            print('MOVING TO QUASIPARTICLE BASIS')
            dnqp_k1, dnqp_k2 = self.quasiparticleBasis()
            self.dnqp_k1 = dnqp_k1
            self.dnqp_k2 = dnqp_k2
            self.dnqp_k1_sorted  = self.splitIntoPulses(self.dnqp_k1)
            self.dnqp_k2_sorted  = self.splitIntoPulses(self.dnqp_k2)
        # this class is now "done..." can modify Elizabeth's class to process whole sweeps. 
        ####################################################################################
        ####################################################################################
    
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
            If True, updates the class attributes like rIQ, time_s, and fs (default: True).
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
            self.rIQ = data_dec
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
    def fitData(self, tunefile=None, band=None, db_offset=None, plot=True):
        """
        load info from self.tune_file, takes caltech style fit. 
        adds attributes fit_dict and fit_errs as outputs from the fit. 
        """
        if not tunefile: 
            tunefile = self.tunefile
        if not band: 
            band = self.band 
        if not db_offset:
            db_offset = self.db_offset 
        ##loads info from the tune file.
        tune_dict= self.loadTuneFile(tunefile=tunefile, band=band,db_offset=db_offset)
        
        ##fitTuneFile returns two dictionaries, one with fit values and the other with errors. 
        self.fit_dict, self.fit_errs = self.fitTuneFile(tune_dict, plot=plot)
        print("Fit dictionary parameters:  ", self.fit_dict)
        self.tune_dict = tune_dict
        return tune_dict 
    def fitTuneFile(self, tune_dict=None, plot=True):
        if not tune_dict: 
            tune_dict = self.tune_dict
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
            charzs.append(chavg[0] + 1j*chavg[1])
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
            chunk_duration_s = int(self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes']))
    
        n_samp = len(data)
    
        # Calculate the number of samples per chunk
        samples_per_chunk = int(fs_Hz * chunk_duration_s)
        num_chunks = n_samp // samples_per_chunk 
        #remove the end of the data, so you can have divisible chunks
        data_chunks = data[start:num_chunks*samples_per_chunk+start]
    
        # Reshape the array into chunks
        data_chunks = data_chunks.reshape(num_chunks, (samples_per_chunk))
        
        return data_chunks 
    def getBigIdx(self, iq_chunks, num_pulses=None, update_params=True):
        if not num_pulses:
            num_pulses = len(self.file_dict['awg_settings']['amplitudes'])
            
        maxs = np.max(np.abs(iq_chunks.real[0:num_pulses]), axis = 1)
        big_idx= np.argmax(maxs)
        if update_params == True: 
            self.bigIdx = big_idx 
        return big_idx
    def getBigIdxAlignmentData(self, alignment_data=None, fs_Hz=None, period_s=None,num_pulses=None, start=0):
        """
        The purpose of this function is to find the bigIdx once, from data where you know  it is a peak-pulse
        """
        try: 
            if not alignment_data:
                aligment_data = self.ideal_I
        except: print('alignment data given')
        if not fs_Hz:
            fs_Hz = self.fs
        if not num_pulses: 
            num_pulses = len(self.file_dict['awg_settings']['amplitudes'])
        if not period_s:
            period_s = self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
            
        alignment_chunks = self.chunkyData(alignment_data, fs_Hz=fs_Hz, chunk_duration_s=period_s, start=start)
        ## Find the chunk which has the highest amplitude pulse. this function only looks at real part of given data.
        print('shape', np.shape(alignment_chunks))
        self.bigIdx = self.getBigIdx(alignment_chunks, num_pulses)
        return   
    def makeShiftedChunks(self, streaming_data, bigIdx=None, fs_Hz=None, period_s=None, num_pulses=None, pre_trigger_s=None, peak=True):
        """
        This function will take flat streaming data, and turn it into a chunked data set.
        The first row will correspond to the "big" pulse in the sequence 
        The pulse will be aligned to start after the pre_trigger_s.
        it will add the atribute: self.shift_idxs_2_cut
        
        Parameters:
        ----------- 
        streaming_data: flat array, data you want to chunk and shift
        bigIdx: int,  what chunk we would originally expect the big pulse to be in (bigIdx=0 means the data started with the biggest pulse)
        fs_Hz: float, sample frequency in Hz
        period_s: float, period between each pulse in the sequence, s
        num_pulses: int, how many pulses were in the pulse sequence? 
        pre_trigger_s: what time in your window do you want the pulse to fall
        peak: Bool. T:  the pulse a peak, F: peak is a dip.

        Returns:
        ----------- 
        end_chunks: a 2D array, with rows corresponding to single pulses, 
        located at pre_trigger_s in the period_s window. The Biggest pulse of the sequence
        will be the first pulse. Will cut first 5 pulses
        if more 'run up' is needed to situate the big pulse correctly. 
        """
        if bigIdx == None:
            bigIdx = self.bigIdx
        if not fs_Hz:
            fs_Hz = self.fs 
        if not num_pulses: 
            num_pulses = len(self.file_dict['awg_settings']['amplitudes'])
        if not period_s:
            period_s = self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
        if not pre_trigger_s: 
            pre_trigger_s = self.pre_trigger_s 
            
        pre_trigger = int(pre_trigger_s  *  fs_Hz ) #s * fs_Hz = samples 
        chunk_len =  int(period_s * fs_Hz )
        
        ##  make sure we dont start on the first pulse in the whole dataset
        if bigIdx == 0: 
            p_start = num_pulses 
        else: p_start = 0
        print(f'cutting {p_start} pulses')
        
        ## This finds the location of the big pulse in the first (num_pulses) 
        if peak: big_pulse_loc  = np.argmax(streaming_data[:chunk_len*(num_pulses)].real)
        else: big_pulse_loc  = np.argmin(streaming_data[chunk_len:chunk_len*(num_pulses)].real)
        ## This cuts the first part of the data, so the chunked data will have big pulse starting in correct point. 
        streaming_start = streaming_data[int((p_start*chunk_len)+big_pulse_loc - pre_trigger): ]
        ##This chunks the data, where the first chunk always starts 
        end_chunks =  self.chunkyData(streaming_start, fs_Hz=fs_Hz , chunk_duration_s=period_s,  start=0)
        self.shift_idxs_2_cut = int((p_start*chunk_len)+big_pulse_loc - pre_trigger)
        return end_chunks     
    def shiftChunks(self, streaming_data, shift_idxs_2_cut=None,  fs_Hz=None, period_s=None):
        """
        purpose of this function is to cut a known number of entries (shift_idx_2_cut)
        off the front of a flattened array (streaaming_data), then chunk it into a 2D array, 
        with length of each row being fs_Hz * period_s.
        needs to be run after "makeShiftedChunks" or provide a shift_idxs_2_cut. 
        Returns the 2D array. 
        """
        if not shift_idxs_2_cut:
            shift_idxs_2_cut = self.shift_idxs_2_cut
        if not fs_Hz:
            fs_Hz = self.fs 
        if not period_s:
            period_s = self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
            
        ##This chunks the data, where the first chunk always starts 
        streaming_start = streaming_data[shift_idxs_2_cut: ]
        end_chunks =  self.chunkyData(streaming_start, fs_Hz=fs_Hz , chunk_duration_s=period_s, start=0)
        return end_chunks
        
    def alignChunks(self, streaming_chunks=None, alignment_chunks=None, total_trigger=None, pre_trigger=None, num_pulses=None, peak=True):
        try:
            if not streaming_chunks: 
                streaming_chunks = self.makeShiftedChunks(self.rIQ)
        except:print('streaming chunks given')
        try: 
            if not alignment_chunks:
                alignment_chunks = self.makeShiftedChunks(self.rIQ.imag)
        except: print('alignment chunks given')
        if not total_trigger: 
            period_s = self.file_dict['awg_settings']['wvfm_duration_s']/len(self.file_dict['awg_settings']['amplitudes'])
            total_trigger = int(period_s * self.fs) #s * fs_Hz = samples 
        if not pre_trigger: 
            pre_trigger = self.pre_trigger_s *  self.fs #s * fs_Hz = samples 
        if not num_pulses: 
            num_pulses = len(self.file_dict['awg_settings']['amplitudes'])
                
        final_chunks = np.zeros([len(alignment_chunks), total_trigger], dtype='complex')

        if peak: fit = fitMaxs(alignment_chunks, num_pulses, pulses_to_fit_shift=[0])
        else: fit = fitMaxs(alignment_chunks, num_pulses, pulses_to_fit_shift=[0])
            
        rows = np.arange(len(alignment_chunks))
        ## Fit returns a linear function st fit(x) = p1x + p2. So this will be applied to each chunk.
        ## Mins gives the location of the pulse according to the fit for each chunk.
        mins = fit(rows)
        for r in rows:
            ##start is a point pre_trigger before the minimum
            start = -1*int(pre_trigger - mins[r])
            ## just finds end so window is of the length you want.
            end = start + total_trigger
            ## This checks if start = end...?
            if len(alignment_chunks[r][start:end]) == 0: 
                print('start', start)
                print('end', end)
                print('mins', mins)
                print('pre trigger', pre_trigger)
            final_chunks[r] = streaming_chunks[r][start:end]
            
        return final_chunks    

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
            mean = np.mean(noise_timestream[:,avg_start_idx:avg_stop_idx], axis=-1, keepdims=True)
        elif len(np.shape(noise_timestream))==1:
            mean = np.mean(noise_timestream[avg_start_idx:avg_stop_idx], axis=-1, keepdims=True)
            
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
            mean = np.mean(timestream_rotated_ideal[:,avg_start_idx:avg_stop_idx], axis=-1, keepdims=True)
        elif len(np.shape(timestream_rotated_ideal)) == 1:
            mean = np.mean(timestream_rotated_ideal[avg_start_idx:avg_stop_idx], axis=-1, keepdims=True)
            
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
        
    def resonator_basis(self, fine_pars, noise_timestream,readout_f,VNA_f,VNA_z,char_f,char_z,  plot_title=None,verbose=True):
    
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
            fig = self.plot_noise_and_vna(timestream_rotated_ideal,MKID_z_ideal,
                                   fit_z=fit_z_ideal, f_idx=res_f_idx,alpha=.05,
                                   char_zs=char_z_rotated_ideal,title='ideal space',verbose=False)
            #fig = self.plot_noise_and_vna(noise_timestream,MKID_z,
            #                       fit_z=fit_z, f_idx=res_f_idx,alpha=.05,
            #                       char_zs=char_z,title='ideal space',verbose=False)
            
        else: fig = plt.figure()
        return frequency, dissipation, ideal, resonator, fig

    def plot_noise_and_vna(self, noise,VNA_z,fit_z=None,f_idx=None,char_zs=None,alpha=0.1,title='',fig_obj=None,save_directory=None,verbose=True):
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
                if idx != 0: label = None
                else: idx = 'arc length direction'
                theta_plot = np.linspace(angles[idx] - dtheta,angles[idx] + dtheta, 100)
                x_plot = radius_mean[idx]*np.cos(theta_plot)
                y_plot = radius_mean[idx]*np.sin(theta_plot)
                plt.plot(x_plot,y_plot,color='k',alpha = 0.6,label=label)
                
                if idx != 0: label = None
                else: idx = 'radius direction'
                radius_plot = np.linspace(0.95*radius_mean[idx],1.05*radius_mean[idx],100)
                x_plot = radius_plot*np.cos(angles[idx])
                y_plot = radius_plot*np.sin(angles[idx])
                plt.plot(x_plot,y_plot,ls='-.',color='k',alpha=0.6,label=label)
    
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

    def quasiparticleBasis(self, frequency=None,dissipation=None,data_T=None,MB_results=None,readout_f=None):
        """
        Parameters:
        -----------
        frequency = flattend or chunked array of frequecy data (df/f)
        dissipation = flattened or chunked array of dissipation data (d(1/Qi))
        data_T = float, temperature in K at which freq/diss data was taken 
        MB_results = an iterable... [f0 (not used), delta (meV), alpha (fractional), Qi0 (unitless)] from MB_fitter. 
        readout_f = float, or iterable,  readout frequency in Hz at which the freq/dissipation data was taken

        Returns:
        -----------
        dnqp_k1: same dim of dissipation 
        dnqp_k2: same dim of frequency
        """
        try:
            if not frequency: 
                frequency = self.dff
        except: print("df/f data given")
        try: 
            if not dissipation: 
                dissipation = self.dis
        except: print('d(1/Qi) data given')
        try:
            if not data_T: 
                data_T = self.data_temp #K
        except: print('temperature data given')
        try: 
            if not MB_results: 
                MB_results = self.MB_results
        except: print("MB results given")
        if not readout_f: 
            readout_f = self.file_dict['freq_in_Hz']

        #MB_f0    = MB_results[0]*1e3    ## MHz
        MB_Delta = MB_results[1]        ## meV
        MB_alpha = MB_results[2]
        MB_Qi0   = MB_results[3]
    
        k1 = MBe.kappa_1(data_T, readout_f, MB_Delta*1e3)*1e6*1e6*1e6 # um^3
        print(f'k1={k1} um3')
        k2 = MBe.kappa_2(data_T, readout_f, MB_Delta*1e3)*1e6*1e6*1e6
        print(f'k2={k2} um3')
 
    
    
        dnqp_k1 = dissipation/(MB_alpha*k1)
        dnqp_k2 = 2*frequency/(MB_alpha*k2)
    
        return dnqp_k1, dnqp_k2 
    def plot_chunks(self,  chunks,pre_trigger=None, show_plot=True, title=None):
        fig,axs = plt.subplots(dpi=300)
        if not pre_trigger:
            pre_trigger = self.pre_trigger_s *  self.fs #s * fs_Hz = samples 
        plt.imshow(chunks)
        plt.vlines(pre_trigger, 0, len(chunks), color='r') #*(1/fs_MHz)*1e-3
        plt.title(title)
        if show_plot: 
            plt.show()
        return fig, axs

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

        
def getFileDict(filename, log_file,):
    """
    This takes in the name of a file. It will find the corresponding entry in the log file and then
    returns a dictionary which will include:
    - band: band data was taken on.
    - channels: channels data was taken on. 
    - filename: the filename of the data.
    - nsamp: the number of samples collected.
    - freq_in_Hz: the tone of the readout frequency.
    - eta_phase_radian: ---ask
    - fs: the sample rate
    - att_uc: attenuation
    - att_dc: dc attenuation
    - period_us: the period of the pulse sequence in us
    - tunefile: the tune file
    - awg_settings: a dictionary of moku settings with
        - power
        ??????????????????
    -led_settings:
    -MEMS: the MEMS settings
        """
    read_log = open(log_file, 'r')
    file_search= filename.split('/')[-1]
    file_search = file_search.split('.')[0]
    done=False
    found=False
    text_dict=''
    print("JSON DICT FOR " + file_search)
    for l_no, line in enumerate(read_log):
        if found and not done:
            text_dict=line
            done = True
        elif "JSON DICT FOR " + file_search in line:
                found=True
    startIndex = text_dict.find("{")
    file_dict = json.loads(text_dict[startIndex:])
        
    return file_dict
 
    