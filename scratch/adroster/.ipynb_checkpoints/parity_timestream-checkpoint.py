""" parity_timestream.py

This module defines a class *ParityTimestream* which we use to analyze timestream data from a variety of readout
techniques on SQUAT detectors.

Authors:
Sam Condon, Hannah Magoon

Date:
2024-10-05
"""
import os
import sys
from turtle import color
import numpy as np
import scipy as sp
import h5py
from scipy.optimize import curve_fit
from scipy.signal import periodogram as psd
from scipy.stats import poisson
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.mixture as mix
import builtins  ## [HM] workaround for us overwriting a builtin function
from qubit_color_dict import qubit_colors

sys.path.append('../Libraries')
from VNA_import_funcs import *

from SMURF_import_funcs import *

## Template functions to fit our data to
def lorentzian_psd(f, S0, fc):
    """ PSD lorentzian function to fit to
    - f: frequency x-axis
    - S0: zero frequency amplitude
    - fc: cutoff frequency
    """
    return S0 / (1 + (f / fc)**2)

def corrected_lorentzian_psd(f, S0, F, Gamma_p, delta_t):
    """
    Lorentzian PSD with correction factor from  eq.14 of [https://arxiv.org/pdf/2402.15471]
    ----------------
    Args:
        f: frequency x-axis
        S0: zero frequency amplitude
        F: parity sequence mapping fidelity
        Gamma_p: parity switching rate
        delta_t: time between parity measurements
    Returns:
        Lorentzian function in frequency space
    """
    return S0 * ((4 * F**2 * Gamma_p) / ((2 * Gamma_p)**2 + (2 * np.pi * f)**2) + (1 - F**2) * delta_t)


def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    """ 2d Gaussian function.

    :param x: x axis data.
    :param y: y axis data.
    :param mu_x: Mean of the Gaussian in the x-dimension.
    :param mu_y: Mean of the Gaussian in the y dimension.
    :param sigma_x: Standard deviation in the x dimension.
    :param sigma_y: Standard deviation in the y dimension.
    :param rho: X/y correlation of the Gaussian.
    :return:
    """
    X, Y = np.meshgrid(x, y)
    Z = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))) * \
        np.exp(-1 / (2 * (1 - rho ** 2)) * (
                ((X - mu_x) ** 2 / sigma_x ** 2) +
                ((Y - mu_y) ** 2 / sigma_y ** 2) -
                (2 * rho * (X - mu_x) * (Y - mu_y) / (sigma_x * sigma_y))
        ))

    return Z


## Data handling class
class ParityTimestream:
    """ Class to import and analyze SQUAT S21 timestream data from a variety of readout techniques.
    ----------
    Init Args:
    :param filepath: Global path to where the data is stored.
    :param filename: String name of the data file.
    :param DAQ_method: S21 readout technique. One of: ['VNA']
    :param channel_num: SMURF channel number
    :param hist_bins: Number of bins with which histograms of the S21 and rotated S21 timestreams are generated.
    ----------
    Class Attributes:
    :attr metadata: Dictionary containing metadata about the acquisition
    :attr data: Dictionary containing the raw data
    :attr analysis: Dictionary containing the results of analysis TO BE WRITTEN TO FILE!
        This includes 'psd_freqs', 'psd_yvals', and filter parameters (not filtered data)
        This also includes tunneling counting data: 'crossings_counts_binned', 'crossings_time_bins_ms', 'crossings_data_axis_label'
    :attr filt_data: Dictionary containing filtered data that should not be written to file
    ----------
    Hidden Class Attributes:
    :attr _vna_data_keys: List of expected keys in the VNA data
    :attr _analysis_prefixes: List of prefixes to identify file contents to be imported into the analysis dictionary
    :attr _qubit_colors: Dictionary of default plot colors for each qubit
    ----------

    """

    # - list of keys storing vna data from vna .npz files
    _vna_data_keys = ['amps', 'phases', 'times_ms']

    # - list of analysis-associated variable prefixes generated during analysis
    _analysis_prefixes = ['psd_freqs', 'savgol', 'moving_avg', 'butter', 'threshold', 'crossings']

    #- Default plot colors for each qubit
    _qubit_colors = qubit_colors

    def __init__(self, filepath, filename, plotpath, DAQ_method='RS_VNA', channel_num = None, hist_bins=100):
        # - Dictionaries hold metadata and data
        self.metadata = {
            'series': 'YYYYMMDD_HHMMSS',
            'filename': None,
            'filepath': None,
            'DAQ_method': 'voodoo',
            'channel_num': None,
            'sample_rate': None,
            'psd_done': False
        }
        self.data = {}

        # - PSD and analysis results that SHOULD be written to file
        self.analysis = {}

        # - filtered data that should never be written to file
        self.filt_data = {'filters_applied': [] }

        # - rotation attributes
        self.IQ_rot_dict = {}
        self.IQ_raw_gmm = mix.GaussianMixture(n_components=2)
        self.IQ_rot_gmm = mix.GaussianMixture(n_components=2)
        self.IQ_phase_gmm = mix.GaussianMixture(n_components=2)

        # - add three component mixtures - #

        self.metadata['filename'] = filename
        self.metadata['filepath'] = filepath
        self.metadata['plotpath'] = plotpath
        self.metadata['DAQ_method'] = DAQ_method
        self.metadata['channel_num'] = channel_num
        if DAQ_method == 'RS_VNA':
            ## Sanity check that it's an npz file
            if filename[-4:] != '.npz':
                raise TypeError('Error: not an npz file, so probably not a VNA file')

            vna_dict = read_file(filepath, filename)
            for key, val in vna_dict.items():
                if key in self._vna_data_keys:
                    self.data[key] = val
                elif any([key.startswith(prefix) for prefix in self._analysis_prefixes]):
                    self.analysis[key] = val
                else:
                    self.metadata[key] = val
        elif DAQ_method == 'CMT_VNA':
            ## Sanity check that it's an hdf5 file
            filename = os.path.join(filepath, filename)
            if filename[-3:] != '.h5':
                raise TypeError('Error: not an hdf5 file, so probably not a CMT VNA file')
            with h5py.File(filename, "r") as f:
                for key in f.attrs.keys():
                    self.metadata[key] = f.attrs[key]
                if not self.metadata.get('meas_type', 'fscan') == 'timestream':
                    raise TypeError('Error: loaded file is not a timestream measurement')
                _sers = self.metadata['series'] 
                self.data['amps']   = np.array(f["amps"])
                self.data['phases'] = np.array(f["phases"])
                self.data['times_ms']  = np.array(f["times_ms"])
        elif DAQ_method == 'SMURF':
            ## Sanity check that it's a SMURF data file
            if filename[-4:] != '.dat':
                raise TypeError('Error: not a .dat file, so probably not a SMURF data file')

            # Check to ensure channel number has been set
            if channel_num == None:
                raise AttributeError('Error: channel_num = None. SMuRF data requires a channel number. Try passing channel_num = 1 ')
            
            channel_num_str = 'channel'+str(channel_num)
            
            d, h, m = read_file(filename)
            self.metadata['sample_rate'] = get_sample_rate(h)
            
            data_reformated = reformat_data(d)
            self.data['amps'] = data_reformated[channel_num_str]['amps']
            self.data['phases'] = data_reformated[channel_num_str]['phases']
            self.data['times_ms'] = match_times(d, h) * 1e3
        
        else:
            raise TypeError('Error: DAQ method not recognized')

        # - compute linear amplitude - #
        amps = self.data['amps']
        amps_lin = 10 ** (amps / 20)
        self.data['Amp_raw'] = amps_lin

        # - compute IQ, IQ_rot, and IQ_phase - #
        phases = self.data['phases']
        self.data['Phase_raw'] = phases
        self.data['I_raw'], self.data['Q_raw'] = amps_lin * np.cos(phases), amps_lin * np.sin(phases)
        iq_rot_dict = self.compute_iq_rot(self.data['I_raw'], self.data['Q_raw'])
        self.data.update(iq_rot_dict)
        iq_phase_dict = self.compute_iq_phase(self.data['I_rot'], self.data['Q_rot'])
        self.data.update(iq_phase_dict)

        # - fit the IQ_rot and IQ_phase Gaussian mixtures. The IQ mixture is fit in compute_iq_rot() - #
        # - IQ_rot
        i_rot, q_rot = self.data['I_rot'], self.data['Q_rot']
        iq_vect = np.zeros((i_rot.shape[0], 2))
        iq_vect[:, 0] = i_rot
        iq_vect[:, 1] = q_rot
        self.IQ_rot_gmm.fit(iq_vect)
        # - IQ_phase
        i_phase, q_phase = self.data['I_phase'], self.data['Q_phase']
        iq_vect = np.zeros((i_phase.shape[0], 2))
        iq_vect[:, 0] = i_phase
        iq_vect[:, 1] = q_phase
        self.IQ_phase_gmm.fit(iq_vect)

        # - compute rot and phase basis amplitude and phase - #
        amp_rot = np.sqrt(np.power(i_rot, 2) + np.power(q_rot, 2))
        phase_rot = np.arctan2(q_rot, i_rot)
        amp_phase = np.sqrt(np.power(i_phase, 2) + np.power(q_phase, 2))
        phase_phase = np.arctan2(q_phase, i_phase)
        self.data.update({
            'Amp_rot': amp_rot,
            'Phase_rot': phase_rot,
            'Amp_phase': amp_phase,
            'Phase_phase': phase_phase,
        })

        # - save the centers of the Gaussian mixtures to variables - #
        self.iq_raw_gaussian_centers = self.IQ_raw_gmm.means_
        self.iq_rot_gaussian_centers = self.IQ_rot_gmm.means_
        self.iq_phase_gaussian_centers = self.IQ_phase_gmm.means_

    # ---------------------------------------- #
    # ------ Non-analysis function ----------- #
    # ---------------------------------------- #

    def __str__(self):
        print('-'*20)
        print('  ' + self.metadata['series'])
        for key in self.metadata.keys():
            if isinstance(self.metadata[key], np.ndarray):
                prinfitt(key + ': array of length ' + str(self.metadata[key].shape))
            else:
                print(key + ': ' + str(self.metadata[key]))
        print('-'*20)
        return ' '

    @property
    def iq_raw_gmm_center(self):
        """ Center (mean values) of the 2-component gaussian mixture model fit to the raw data.

        :return: Pandas dataframe with row corresponding to the gaussian mixture component and columns ["I", "Q"]
        """
        means = self.IQ_raw_gmm.means_
        df = pd.DataFrame(means, columns=['I', 'Q'])
        return df

    @property
    def iq_rot_gmm_center(self):
        """ Center (mean values) of the 2-component gaussian mixture model fit to the rotated data.

        :return: Pandas dataframe with row corresponding to the gaussian mixture component and columns ["I", "Q"]
        """
        means = self.IQ_rot_gmm.means_
        df = pd.DataFrame(means, columns=['I', 'Q'])
        return df

    @property
    def iq_phase_gmm_center(self):
        """ Center (mean values) of the 2-component gaussian mixture model fit to the phase basis data.

        :return: Pandas dataframe with row corresponding to the gaussian mixture component and columns ["I", "Q"]
        """
        means = self.IQ_phase_gmm.means_
        df = pd.DataFrame(means, columns=['I', 'Q'])
        return df

    def get_metadata_string(self):
        ## [HM 10/9/2024] updating this so that it doesn't break if metadata is missing
        """ Make a string summarizing metadata for plot titles
        ----------
        Returns:
            metadata_string: str
        """
        if self.metadata['DAQ_method'] == 'RS_VNA':
            meta_str0 = '; '.join([
                self.metadata.get('series', 'N/A'),
                f'DAQ = {self.metadata.get("DAQ_method", "N/A")}',
                f'{self.metadata.get("device", "N/A")}_{self.metadata.get("qubit", "N/A")}',
                f'Freq = {(self.metadata.get("cw_freq_mhz") * 1e-3) if self.metadata.get("cw_freq_mhz") is not None else "N/A"} GHz',
                f'IFBW = {self.metadata.get("bandwidth", "N/A")} Hz',
            ])

            meta_str1 = '; '.join([
                f'Warm atten = {self.metadata.get("warm_att", "N/A")} dB',
                f'Cold atten = {self.metadata.get("cold_att", "N/A")} dB',
                f'Power = {self.metadata.get("vna_power", "N/A")} dBm',
                f'Power at device = {self.metadata.get("power_at_device", "N/A")} dBm',
            ])
            metadata_string = '\n'.join([meta_str0, meta_str1])
        elif self.metadata['DAQ_method'] == 'CMT_VNA':
            meta_str0 = '; '.join([
                self.metadata.get('series', 'N/A'),
                f'DAQ = {self.metadata.get("DAQ_method", "N/A")}',
                f'{self.metadata.get("device", "N/A")}_{self.metadata.get("qubit", "N/A")}',
                f'Freq = {float(self.metadata.get("f_cw")) * 1e-9 if self.metadata.get("f_cw") is not None else "N/A"} GHz',
                f'IFBW = {self.metadata.get("ifbw", "N/A")} Hz',
            ])

            meta_str1 = '; '.join([
                f'Power = {self.metadata.get("vna_power", "N/A")} dBm',
                f'Power at device = {self.metadata.get("device_power", "N/A")} dBm',            
            ])
            metadata_string = '\n'.join([meta_str0, meta_str1])
        
        elif self.metadata['DAQ_method'] == 'SMURF':
            metadata_string = self.metadata['DAQ_method'] + ' channel' + str(self.metadata['channel_num'])
            
        return metadata_string

    def write_to_file(self, expt_path=None, verbose=True):
        """ Writes the metadata, data, and analysis dictionaries to a file
        ----------
            Args:
            expt_path: str
                Path to save the file. If None, then pulls from the metadata
            verbose: bool
                If True, prints out the file path
        """
        if expt_path is None: expt_path = self.metadata['filepath']
        if not self.metadata['psd_done']:
            print('Error: PSD not calculated yet')
            return None
        if self.metadata['DAQ_method'] == 'VNA':
            big_dict = {**self.metadata, **self.data, **self.analysis}
            write_file(big_dict, filepath=expt_path, filename=self.metadata['filename'][16:], verbose=verbose)
        return

    # ---------------------------------------- #
    # -------- Mega Function Calls ----------- #
    # ---------------------------------------- #

    # once we decide on a rotation scheme, will set up a function to rotate the data,
    # run the filter, and plot a histogram

    # ---------------------------------------- #
    # -------- Rotation functions ------------ #
    # ---------------------------------------- #

    @staticmethod
    def rotate_iq(i, q, angle):
        Irot = i * np.cos(angle) - q * np.sin(angle)
        Qrot = i * np.sin(angle) + q * np.cos(angle)

        return Irot, Qrot

    def compute_iq_rot(self, i, q):
        """ Compute I and Q in the IQ_rot basis.

        :param i: Raw I data vector.
        :param q: Raw Q data vector.
        :return: Dictionary with rotated I and Q data.
        """
        ret_dict = {}

        # - fit 2 component gaussian mixture model to the 2d IQ data
        iq_vect = np.zeros((i.shape[0], 2))
        iq_vect[:, 0] = i
        iq_vect[:, 1] = q
        self.IQ_raw_gmm.fit(iq_vect)
        imeans, qmeans = self.IQ_raw_gmm.means_.transpose()
        i0_center, i1_center = imeans
        q0_center, q1_center = qmeans
        rot_i = i1_center - i0_center
        rot_q = q1_center - q0_center

        # - determine the rotation angle, account for the quadrant
        rot_angle = -1*np.arctan2(rot_q, rot_i)
        # - rotate into the IQ_rot basis, remove the mean offset in the vertical axis - #
        i_rot, q_rot = self.rotate_iq(i, q, rot_angle)
        q_rot -= q_rot.mean()

        # - fit a temporary gaussian mixture, translate on the horizontal axis to place mixture centers symmetric about
        # - the vertical axis.
        gmm_tmp = mix.GaussianMixture(n_components=2)
        iq_vect = np.zeros((i_rot.shape[0], 2))
        iq_vect[:, 0] = i_rot
        iq_vect[:, 1] = q_rot
        gmm_tmp.fit(iq_vect)
        imeans, qmeans = gmm_tmp.means_.transpose()
        i_rot -= imeans.mean()

        ret_dict['I_rot'] = i_rot
        ret_dict['Q_rot'] = q_rot

        return ret_dict

    def compute_iq_phase(self, i_rot, q_rot, gmm=None):
        """ Generate the IQ_phase basis data from the IQ_rot data computed with `compute_iq_rot()`

        :param i_rot:
        :param q_rot:
        :param gmm: Gaussian mixture model object used to compute the IQ_rot basis data.
        :return: Dictionary with the I_phase and Q_phase data.
        """
        # - calculate the distance between the centers of the Gaussian mixture fit centers - #
        if gmm is None:
            gmm = self.IQ_raw_gmm
        imeans, qmeans = gmm.means_.transpose()
        dist = ((imeans[1] - imeans[0])**2 + (qmeans[1] - qmeans[0])**2)**(1/2)

        # - rotate the data by pi/2, translate in I by the radius of the circle containing the Gaussian mixture centers
        i_phase, q_phase = self.rotate_iq(i_rot, q_rot, np.pi / 2)
        i_phase += dist / 2

        # - fit the gmm - #
        iq_vect = np.zeros((i_phase.shape[0], 2))
        iq_vect[:, 0] = i_phase
        iq_vect[:, 1] = q_phase
        self.IQ_phase_gmm.fit(iq_vect)

        ret_dict = {
            'I_phase': i_phase,
            'Q_phase': q_phase,
        }

        return ret_dict

    # ---------------------------------------- #
    # -------- Filtering functions ----------- #
    # ---------------------------------------- #

    def savgol_filter(self, window=50, poly=10, data_axis_label='Amps_raw', verbose=True):
        """ Apply a Savitzky-Golay filter to the data
        ----------
        Args:
            window: int
                Window length for the filter
            poly: int
                Polynomial order for the filter
            data_axis_label: str
                Key in the data dictionary indicating the data to filter
        """
        ## Check if the data is in the dictionary
        if data_axis_label not in self.data.keys():
            print('Error: axis label not found in data dictionary')
            return None

        ## Apply the filter
        data_name = 'savgol_'+ data_axis_label
        if verbose: print('Savgol filtered data stored as:', data_name)
        self.filt_data[data_name] = sp.signal.savgol_filter(self.data[data_axis_label], window, poly)

        ## Update the analysis dictionary with filter parameters
        self.analysis['savgol_window'] = window
        self.analysis['savgol_poly'] = poly
        self.analysis['savgol_axis'] = data_axis_label
        if 'savgol' not in self.filt_data['filters_applied']:
            self.filt_data['filters_applied'].append('savgol')
        return

    def moving_average_filter(self, window, data_axis_label='Amps_raw', verbose=True):
        """ Apply a moving average filter to the data
        ----------
        Args:
            window: int
                Window length for the filter
            data_axis_label: str
                Key in the data dictionary indicating the data to filter
        """
        ## Check if the data is in the dictionary
        if data_axis_label not in self.data.keys():
            print('Error: axis label not found in data dictionary')
            return None

        ## Apply the filter
        data_name = 'moving_avg_' + data_axis_label
        if verbose: print('Moving average filtered data stored as:', data_name)
        self.filt_data[data_name] = np.convolve(self.data[data_axis_label], np.ones(window)/window, mode='same')

        ## Update the analysis dictionary with filter parameters
        self.analysis['moving_avg_window'] = window
        self.analysis['moving_avg_axis'] = data_axis_label
        if 'moving_avg' not in self.filt_data['filters_applied']:
            self.filt_data['filters_applied'].append('moving_avg')
        return

    def butterworth_filter(self, cutoff_hz=80, order=1, data_axis_label='Amps_raw', verbose=True):
        """ Apply a Butterworth filter to the data
        ----------
        Args:
            cutoff_hz: float
                Cutoff frequency for the filter [Hz]
            order: int
                Order of the filter
            data_axis_label: str
                Key in the data dictionary indicating the data to filter
        """
        ## Check if the data is in the dictionary
        if data_axis_label not in self.data.keys():
            print('Error: axis label not found in data dictionary')
            return None

        ## Apply the filter
        data_name = 'butter_' + data_axis_label
        if verbose: print('Butterworth filtered data stored as:', data_name)
        time_data_sec = self.data['times_ms'] / 1e3
        b, a = sp.signal.butter(N=order, Wn=cutoff_hz, btype='low', fs=1/(time_data_sec[1] - time_data_sec[0]))
        zi = sp.signal.lfilter_zi(b, a)
        filt_data, _ = sp.signal.lfilter(b, a, self.data[data_axis_label], zi=zi*self.data[data_axis_label][0])
        self.filt_data[data_name] = filt_data

        ## Update the analysis dictionary with filter parameters
        self.analysis['butter_cutoff_hz'] = cutoff_hz
        self.analysis['butter_order'] = order
        self.analysis['butter_axis'] = data_axis_label
        if 'butter' not in self.filt_data['filters_applied']:
            self.filt_data['filters_applied'].append('butter')
        return

    # ---------------------------------------- #
    # -------- PSD functions ----------------- #
    # ---------------------------------------- #

    def calc_psd(self, data_axis_label, nfft=2 ** 12):
        """ Calculates the power spectral density of the data stored in the class
        ----------
        Args:
            data_axis_label: str
                Key in the data dictionary to the data to take the PSD of
            nfft: int
                Number of points to use in the FFT
        """
        ## check that axis label is in the data dictionary
        if data_axis_label not in self.data.keys():
            print('Error: axis label not found in data dictionary')
            return None
        
        ## Normalize the data in the time domain
        centerline = self._calc_threshold(data_axis_label)
        tel_state1 = np.mean(self.data[data_axis_label][self.data[data_axis_label] < centerline])
        tel_state2 = np.mean(self.data[data_axis_label][self.data[data_axis_label] > centerline])
        tel_separation = np.abs(tel_state2 - tel_state1)
        datavals = (self.data[data_axis_label] - centerline) / tel_separation
        
        ## Calculate the PSD and save it to the class
        times_sec = self.data['times_ms'] / 1e3
        dt = times_sec[1] - times_sec[0]
        f, p = psd(datavals, fs=1 / dt, nfft=nfft)
        self.metadata['psd_done'] = True
        self.metadata['psd_axis'] = data_axis_label
        self.analysis['psd_freqs'] = f
        self.analysis['psd_yvals'] = p
        self.metadata['n_files'] = 1
        self.analysis['tel_cen'] = centerline
        self.analysis['tel_sep'] = tel_separation
        return

    #- Fits a lorentzian to the PSD data stored in the class
    
    def fit_psd(self, fit_to_corrected_function=True, min_freq_bound=None):
        """ Fits a lorentzian to the PSD data stored in the class.
            Stores the fit parameters in the analysis dictionary.
        ----------
            Args:
                fit_to_corrected_function: bool
                    If True, fits the lorentzian function with noise floor correction
                    If False, fits the standard lorentzian function to the PSD data
                min_freq_bound: float
                    Minimum frequency bound [in Hz] for the fit.  If None, then no bound is set 
        """        
        if not self.metadata['psd_done']:
            print('Error: PSD not calculated yet')
            return None
                
        try:
            ## Set initial guesses based on the maximum point in the data
            S0_guess = np.max(self.analysis['psd_yvals'])
            peak_freq = self.analysis['psd_freqs'][np.argmax(self.analysis['psd_yvals'])]
            freq_bound = 0 if min_freq_bound is None else min_freq_bound
            
            if fit_to_corrected_function:
                F_guess = 0.9
                delta_t = (self.data['times_ms'][1] - self.data['times_ms'][0]) / 1e3
                
                if self.metadata['DAQ_method'] == 'SMURF':
                    sample_rate_hz = self.metadata['sample_rate']
                    delta_t = 1 / sample_rate_hz
                
                ## Fit with corrected Lorentzian function
                popt, pcov = curve_fit(
                    lambda f, S0, F, Gamma_p: corrected_lorentzian_psd(f, S0, F, Gamma_p, delta_t),
                    self.analysis['psd_freqs'][1:], 
                    self.analysis['psd_yvals'][1:],
                    p0=[S0_guess, F_guess, peak_freq],
                    sigma = abs(self.analysis['psd_yvals'][1:])/np.sqrt(self.metadata['n_files']),
                    absolute_sigma=True,
                    bounds=([-np.inf, 0, freq_bound], [np.inf, 1, np.inf])
                )
                fit_params = {
                    'psd_corr_fit_S0': popt[0],
                    'psd_corr_fit_F': popt[1],
                    'psd_corr_fit_Gamma_p': popt[2],
                    'psd_corr_fit_delta_t': delta_t,
                    'psd_corr_fit_pcov': pcov
                }
                
            else:
                ## Fit with standard Lorentzian function
                popt, pcov = curve_fit(
                    lorentzian_psd,
                    self.analysis['psd_freqs'][1:], 
                    self.analysis['psd_yvals'][1:],
                    p0=[S0_guess, peak_freq],
                    sigma = abs(self.analysis['psd_yvals'][1:])/np.sqrt(self.metadata['n_files']),
                    absolute_sigma=True,
                    bounds=([-np.inf, freq_bound], [np.inf, np.inf])
                )
                fit_params = {
                    'psd_fit_S0': popt[0],
                    'psd_fit_fc': popt[1],
                    'psd_fit_pcov': pcov
                }

            ## Store fit parameters in self.analysis dictionary
            for key, val in fit_params.items():
                self.analysis[key] = val

        except Exception as e:
            ## Print error message, set fit_params to None, and store None in analysis keys
            print(f"Error during fitting: {e}")
            fit_params = None
            fit_keys = ['psd_corr_fit_S0', 'psd_corr_fit_F', 'psd_corr_fit_Gamma_p', 'psd_corr_fit_pcov'] if fit_to_corrected_function else ['psd_fit_S0', 'psd_fit_fc', 'psd_fit_pcov']
            for key in fit_keys:
                self.analysis[key] = None
        return fit_params

    ## ---------------------------------------- ##
    ## ------- Event tagging functions -------- ##
    ## ---------------------------------------- ##
    def _calc_threshold(self, data_axis_label):
        """ Calculates the threshold for event tagging
        ----------
        Args:
            data_axis_label: str
                Name of the data axis to calculate the threshold for
        """
        ## Look in both the data and filtered data dictionaries
        if data_axis_label in self.data.keys():
            self.analysis['threshold'] = np.mean(self.data[data_axis_label])
        elif data_axis_label in self.filt_data.keys():
            self.analysis['threshold'] = np.mean(self.filt_data[data_axis_label])
        else:
            print('Error: axis label not found in data or filt_data dictionaries')
            return None
        self.analysis['threshold_axis'] = data_axis_label
        return self.analysis['threshold']

    def count_events(self, data_axis_label, time_bin_ms=10, verbose=True, plot=True, xaxis_seconds=1, plot_all=True):
        """ Counts the number of threshold crossings, stores the counts in the analysis dictionary
        ----------
        Args:
            data_axis_label: str
                Name of the data axis to calculate the threshold for
            time_bin_ms: int
                Time bin width for counting events
            verbose: bool
                If True, print debug information
            plot: bool
                If True, plots the data and crossing locations
            xaxis_seconds: int
                Number of seconds to plot on the x-axis
            plot_all: bool
                If True, plots all the data. If False, only plots the first xaxis_seconds
        """
        ## Look in both the data and filtered data dictionaries
        if data_axis_label in self.data.keys():
            data = self.data[data_axis_label]
        elif data_axis_label in self.filt_data.keys():
            data = self.filt_data[data_axis_label]
        else:
            print('Error: axis label not found in data or filt_data dictionaries')
            return None
        threshold = self._calc_threshold(data_axis_label)

        ## Check that the timebin is larger than the time resolution of the data
        if time_bin_ms < self.data['times_ms'][1] - self.data['times_ms'][0]:
            print('Error: time bin too small')
            return None
        if verbose:
            print('Sample rate is', self.data['times_ms'][1] - self.data['times_ms'][0], 'ms')
            print('Time bin width is', time_bin_ms, 'ms, which = ',
                  time_bin_ms / (self.data['times_ms'][1] - self.data['times_ms'][0]), 'samples')

        ## Convert times to seconds
        times_sec = self.data['times_ms'] / 1e3

        ## Find the crossings: sign change of (data - threshold)
        crossings = np.where(np.diff(np.signbit(data - threshold)))[0]

        ## Count number of crossings per time bin
        n_bins = int(np.ceil(times_sec[-1] / (time_bin_ms / 1e3)))
        event_counts = np.zeros(n_bins)
        for i in range(n_bins):
            bin_inds = (times_sec >= i * time_bin_ms / 1e3) & (times_sec < (i + 1) * time_bin_ms / 1e3)
            event_counts[i] = np.sum(np.isin(crossings, np.where(bin_inds)[0]))
        self.analysis['crossings_counts_binned'] = event_counts
        self.analysis['crossings_time_bins_ms'] = np.arange(0, n_bins * time_bin_ms, time_bin_ms)
        self.analysis['crossings_data_axis_label'] = data_axis_label

        ## Plot the data and crossing locations. Split data into plots of xaxis_seconds
        if plot:
            n_subplots = int(np.ceil(times_sec[-1] / xaxis_seconds))
            if not plot_all: n_subplots=1
            for i in range(n_subplots):
                inds = (times_sec >= i * xaxis_seconds) & (times_sec < (i + 1) * xaxis_seconds)
                crossings_in_range = crossings[(times_sec[crossings] >= i * xaxis_seconds) &
                                               (times_sec[crossings] < (i + 1) * xaxis_seconds)]
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(times_sec[inds], data[inds], label='Data', color=self._qubit_colors[self.metadata['device']+self.metadata['qubit']])
                ax.axhline(threshold, color='grey', linestyle='--', label='Threshold')
                ax.scatter(times_sec[crossings_in_range], data[crossings_in_range], color='black', label='Crossings', zorder=5, marker='x')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(data_axis_label)
                ax.legend(loc='upper right')
                fig.suptitle(f"{self.metadata['series']} Threshold crossings for {data_axis_label} (Part {i+1})")
            ret = (self.analysis['crossings_counts_binned'], self.analysis['crossings_time_bins_ms'], fig, ax)
        else:
            ret = (self.analysis['crossings_counts_binned'], self.analysis['crossings_time_bins_ms'])

        return ret

    # ---------------------------------------- #
    # -------- Plotting functions ------------ #
    # ---------------------------------------- #


    def plot_histogram_for_multiple_time_bins(self, data_axis_label, time_bin_ms_list,
                                              plot_title=None, saveplot=True, ax=None):
        """
        Repeatedly calls the count_events function for different time bin sizes and plots the results
        on the same histogram, using the magma colormap with lighter colors for larger time bins.
        ----------
        Args:
            data_axis_label: str
                Name of the data axis used to calculate the threshold for
            time_bin_ms_list: list
                List of time bin sizes (in ms) to try
            plot_title: str, optional
                Title for the plot. If None, then a title is generated from the metadata
            saveplot: bool, optional
                If True, saves the figure is saved to the plotpath in the metadata
            verbose: bool, optional
                If True, print debug information
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
        """
        # - initialize figure and axes objects - #
        fig = None
        if ax is None:
            fig, ax = plt.figure(figsize=(10, 6))
        if plot_title is None:
            plot_title = self.get_metadata_string() + f'\n{data_axis_label}'
        ax.set(xlabel='Time (s)', ylabel='Number of Tunneling Events', title=plot_title)

        ## Get the magma colormap and normalize it
        cmap = cm.get_cmap('magma', len(time_bin_ms_list))
        colors = [cmap(i) for i in range(len(time_bin_ms_list))]

        ## Sort time_bin_ms_list in ascending order (darker for smaller, lighter for larger)
        sorted_indices = np.argsort(time_bin_ms_list)
        time_bin_ms_list_sorted = np.flip(np.array(time_bin_ms_list)[sorted_indices])
        colors_sorted = np.array(colors)[sorted_indices]

        ## Loop through the provided time bin sizes, add to histogram
        #print(colors_sorted)
        print(time_bin_ms_list_sorted)
        for i, (time_bin_ms, color) in enumerate(zip(time_bin_ms_list_sorted, colors_sorted)):
            event_counts, time_bins = self.count_events(data_axis_label, time_bin_ms, verbose=False, plot=False)
            time_bins_sec = time_bins / 1000

            ax.bar(time_bins_sec, event_counts, width=(time_bin_ms / 1000),
                   alpha=0.8, label=f'Time bin: {time_bin_ms} ms', color=color, edgecolor='black')
        ax.legend(loc='upper right')
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_histogram_varying_binwidth_colormap.png'
            ax.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_events_per_bin_for_multiple_time_bins(self, data_axis_label, time_bin_ms_list,
                                                   plot_title=None, saveplot=True, ax=None):
        """
        Repeatedly calls the count_events function for different time bin sizes
        Plots distribution of events per bin for each time bin size
        ----------
        Args:
            data_axis_label: str
                Name of the data axis used to calculate the threshold for
            time_bin_ms_list: list
                List of time bin sizes (in ms) to try
            plot_title: str, optional
                Title for the plot. If None, then a title is generated from the metadata
            saveplot: bool, optional
                If True, saves the figure is saved to the plotpath in the metadata
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.figure(figsize=(10, 6))
        if plot_title is None:
            plot_title = self.get_metadata_string() + f'\n{data_axis_label}'
        ax.set(xlabel='Number of tunneling events in time bin', ylabel='Number of time bins with the given event count',
               title=plot_title)

        cmap = cm.get_cmap('viridis', len(time_bin_ms_list))
        colors = [cmap(i) for i in range(len(time_bin_ms_list))]

        ## Sort time_bin_ms_list in ascending order (darker for smaller, lighter for larger)
        sorted_indices = np.argsort(time_bin_ms_list)
        time_bin_ms_list_sorted = (np.array(time_bin_ms_list)[sorted_indices])
        colors_sorted = np.array(colors)[sorted_indices]
        data_axis_label = self.analysis['crossings_data_axis_label']

        ## Loop through the provided time bin sizes, add to histogram
        for i, (time_bin_ms, color) in enumerate(zip(time_bin_ms_list_sorted, colors_sorted)):
            ## Run analysis and tally event counts
            event_counts, time_bins = self.count_events(data_axis_label, time_bin_ms, verbose=False, plot=False)
            n_events = np.arange(0, np.max(event_counts) + 1)
            n_events_counts = np.zeros(len(n_events))
            for i in range(len(n_events)):
                n_events_counts[i] = np.sum(event_counts == i)
            ax.bar(n_events, n_events_counts, color=color, edgecolor='black', align='center', width=0.95,
                   label=f'Time bin: {time_bin_ms} ms')
        ax.legend(loc='upper right')
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_events_per_bin_varying_binwidth_colormap.png'
            ax.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_psd(self, plot_title=None, ymin=1e-12, ymax=1, savefig=True, ax=None):
        """ Plots the PSD data stored in the class
        ----------
        Args:
            plot_title: str
                Title for the plot. If None, then a title is generated from the metadata
            ymin: float
                Minimum y-axis value
            ymax: float
                Maximum y-axis value
            savefig: bool
                If True, saves the figure to the path saved to the class metadata
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
        """
        if not self.metadata['psd_done']:
            print('Error: PSD not calculated yet')
            return None

        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        auto_plot_title = f"{self.metadata['series']}  -- {self.metadata['psd_axis']}"
        if 'mxc_temp' in self.data.keys():
            auto_plot_title += f"  --  {self.data['mxc_temp']} mK"
        if plot_title is not None:
            auto_plot_title += '\n' + plot_title
        ax.set(xlabel='Frequency (Hz)', ylabel='Power spectral density', title=auto_plot_title)

        ax.loglog(self.analysis['psd_freqs'], self.analysis['psd_yvals'])
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        if savefig:
            if not os.path.exists(self.metadata['plotpath']):
                os.makedirs(self.metadata['plotpath'])
            plot_filename = self.metadata['filename'][:-4] + '_psd.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_psd_fit(self, plot_title=None, ymin=None, ymax=None, savefig=True,
                     plot_corrected_fit=True, plot_standard_fit=True, linear_ax=False, ax=None,
                     hide_axis_text=False, use_qubit_color_code=True):
        """
        Plots the PSD data along with the lorentzian fit
        ----------
        Args:
            plot_title: str
                Title for the plot. If None, then a title is generated from the metadata
            ymin: float
                Minimum y-axis value.  If none, then a ymin is guessed from the data
            ymax: float
                Maximum y-axis value. If none, then a ymax is guessed from the data
            savefig: bool
                If True, saves the figure to the path saved to the class metadata
            plot_corrected_fit: bool
                If True, plots the corrected lorentzian fit
            plot_standard_fit: bool
                If True, plots the standard lorentzian fit
            linear_ax: bool
                If True, plots the y-axis in linear scale
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
            hide_axis_text: bool
                If True, hides the text on plot specifying PSD axis. Use this if you are making presentation plots
            use_qubit_color_code: bool
                If True, uses the qubit color code for the plot.  If False, uses the default color code
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        if plot_title is None:
            plot_title = self.get_metadata_string()
        ax.set(xlabel='Frequency [Hz]', ylabel='Power spectral density', title=plot_title)

        ## Sanity checks
        if not self.metadata['psd_done']:
            print('Error: PSD not calculated yet.  Run self.calc_psd()')
            return None
        if 'psd_fit_S0' not in self.analysis.keys() and 'psd_corr_fit_S0' not in self.analysis.keys():
            print(self.analysis.keys())
            print('Error: PSD not fit yet.  Run self.fit_psd()')
            return None
        if plot_corrected_fit and 'psd_corr_fit_S0' not in self.analysis.keys():
            print('Error: Corrected fit not performed'); plot_corrected_fit = False
        if plot_standard_fit and 'psd_fit_S0' not in self.analysis.keys():
            print('Error: Standard fit not performed'); plot_standard_fit = False
        if plot_corrected_fit and self.analysis['psd_corr_fit_S0'] is None:
            print('Error: Corrected fit failed.  Cannot plot'); plot_corrected_fit = False
        if plot_standard_fit and self.analysis['psd_fit_S0'] is None:
            print('Error: Standard fit failed.  Cannot plot'); plot_standard_fit = False

        ## Configure colors
        try:
            data_color = self._qubit_colors[self.metadata['device'] + self.metadata['qubit']] if use_qubit_color_code else 'steelblue'
        except:
            data_color = 'steelblue'
        fit_color = 'grey' if use_qubit_color_code else 'indigo'
        corrected_fit_color = 'black' if use_qubit_color_code else 'indianred'
        
        if linear_ax:
            ax.semilogx(self.analysis['psd_freqs'], self.analysis['psd_yvals'], label='Data', color=data_color)
            if plot_standard_fit:
                ax.semilogx(self.analysis['psd_freqs'],
                            lorentzian_psd(self.analysis['psd_freqs'], self.analysis['psd_fit_S0'], self.analysis['psd_fit_fc']),
                            label=f'Fc = {self.analysis["psd_fit_fc"]:.2f} Hz', color=fit_color, linewidth=3)
            if plot_corrected_fit:
                delta_t = (self.data['times_ms'][1] - self.data['times_ms'][0]) / 1e3
                ax.semilogx(self.analysis['psd_freqs'],
                            corrected_lorentzian_psd(self.analysis['psd_freqs'], self.analysis['psd_corr_fit_S0'],
                                                     self.analysis['psd_corr_fit_F'], self.analysis['psd_corr_fit_Gamma_p'], delta_t),
                            label=f'Γ_p = {self.analysis["psd_corr_fit_Gamma_p"]:.2f} +- {np.sqrt(np.diag(self.analysis["psd_corr_fit_pcov"])[2]):.2f} Hz', color=corrected_fit_color, linewidth=3)
        else:
            ax.loglog(self.analysis['psd_freqs'], self.analysis['psd_yvals'], label='Data', color=data_color)
            if plot_standard_fit:
                ax.loglog(self.analysis['psd_freqs'],
                          lorentzian_psd(self.analysis['psd_freqs'], self.analysis['psd_fit_S0'], self.analysis['psd_fit_fc']),
                          label=f'Fc = {self.analysis["psd_fit_fc"]:.2f} Hz', color=fit_color, linewidth=3)
            if plot_corrected_fit:
                delta_t = (self.data['times_ms'][1] - self.data['times_ms'][0]) / 1e3
                ax.loglog(self.analysis['psd_freqs'],
                        corrected_lorentzian_psd(self.analysis['psd_freqs'], self.analysis['psd_corr_fit_S0'], 
                                                self.analysis['psd_corr_fit_F'], self.analysis['psd_corr_fit_Gamma_p'], delta_t),
                        label=f'Γ_p = {self.analysis["psd_corr_fit_Gamma_p"]:.2f} +- {np.sqrt(np.diag(self.analysis["psd_corr_fit_pcov"])[2]):.2f} Hz', color=corrected_fit_color, linewidth=3)
                ax.grid()
        if not hide_axis_text:       
            ax.text(0.02, 0.02, f'Axis: {self.metadata["psd_axis"]}', transform=plt.gca().transAxes, fontsize=13, color='gray')
        if plot_title is None:
            plot_title = self.get_metadata_string()
        ax.set_title(plot_title)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(np.min(self.analysis['psd_yvals'][2:-2])*0.1, np.max(self.analysis['psd_yvals'][2:-2])*50)
        ax.legend()
        if savefig:
            plot_filename = self.metadata['filename'][:-4] + '_psd_fit.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))
        return fig, ax

    def plot_1d_timestream_and_hist(self, basis, axes=None, hist_bins=100, axs=None):
        """ Plot a timestream and overlayed histogram of one axis in the desired basis.

        :param basis: Which basis to plot under. Valid values are ['raw', 'rot', 'phase']
        :param axes: String or list-like object of strings specifying which axis/axes to plot. Valid axes are:
        ['I', 'Q', 'Amp', 'Phase']
        :param hist_bins: The number of bins to use in the histogram.
        :param axs: Array of Matplotlib Axes objects. Optional, if provided these objects will be plotted on rather than
        creating new objects.
        """
        # - configure the figure and axes - #
        fig = None
        if axs is None:
            fig = plt.figure()
            fig.suptitle(self.get_metadata_string())
            nrows = 4 if axes is None else len(axes)
            gs = mpl.gridspec.GridSpec(
                figure=fig,
                nrows=nrows, ncols=2,
                wspace=0.02,
                width_ratios=[2 / 3, 1 / 3],
                top=0.9, bottom=0.1,
                left=0.1, right=0.9
            )
            axs = gs.subplots(sharey='row')
            if nrows == 1:
                axs = np.array([axs])
        ts_axs = axs[:, 0].flatten()
        hist_axs = axs[:, 1].flatten()
        ts_axs[-1].set_xlabel('Time (ms)')
        hist_axs[-1].set_xlabel('Counts')
        ylabel_strings = {
            'I': {'raw': r'$\text{I}_{\text{raw}}$', 'rot': r'$\text{I}_{\text{rot}}$',
                  'phase': r'$\text{I}_{\text{phase}}$'},
            'Q': {'raw': r'$\text{Q}_{\text{raw}}$', 'rot': r'$\text{Q}_{\text{rot}}$',
                  'phase': r'$\text{Q}_{\text{phase}}$'},
            'Amp': {'raw': r'$|R|_{\text{raw}}$', 'rot': r'$|R|_{\text{rot}}$', 'phase': r'$|R|_{\text{phase}}$'},
            'Phase': {'raw': r'$\theta_{\text{raw}}$', 'rot': r'$\theta_{\text{rot}}$',
                      'phase': r'$\theta_{\text{phase}}$'},
        }
        unit_strings = {
            'I': '(V)', 'Q': '(V)', 'Amp': '(V)',
            'Phase': '',
        }

        # - get the data - #
        if axes is None:
            axes = ['I', 'Q', 'Amp', 'Phase']
        elif type(axes) is str:
            axes = [axes]
        data_dict = {
            axis: self.data[f'{axis}_{basis}']
            for axis in axes
        }
        time = self.data['times_ms']
        dev = ''.join([self.metadata['device'], self.metadata['qubit']])
        plot_color = self._qubit_colors[dev]

        # - plot the data - #
        for i, ad in enumerate(data_dict.items()):
            axis, data = ad
            ylabel = ' '.join([ylabel_strings[axis][basis], unit_strings[axis]])
            ts_axs[i].set_ylabel(ylabel)
            ts_axs[i].scatter(time, data, marker='.', s=1, color=plot_color)
            hist_axs[i].hist(data, orientation='horizontal', bins=hist_bins, alpha=0.5, color=plot_color)
            hist_axs[i].hist(data, orientation='horizontal', bins=hist_bins, histtype='step', color=plot_color)

        return fig, axs

    def plot_gaussian_mixture(self, basis='raw', bins=100, cmap='viridis', rng=None, axs=None):
        """ Plot IQ data in the chosen basis, and the GaussianMixture model fit side by side.

        :param basis: The basis in which to plot data and Gaussian mixtures from. Should be one of
                      ['raw', 'rot', 'phase']
        :param bins: Number of bins to split the raw IQ data into.
        :param cmap: Colormap string.
        :param rng: Optional axes limits.
        :param axs: Optional array of Matplotlib Axes objects. If provided, these will be
                    plotted on rather than new objects being created.

        :return: Matplotlib figure object (or None if axs is provided) and array of Axes objects
                 post plotting.
        """
        fig = None
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.25))
            fig.suptitle(self.get_metadata_string())
        axs[0].set_title(r'$\text{IQ}_{\text{%s}}$ data' % basis)
        axs[1].set_title('Gaussian Mixture Model Fit')
        for ax in axs:
            ax.set(xlabel=r'$\text{I}_\text{%s}$' % basis, ylabel=r'$\text{Q}_\text{%s}$' % basis)

        # - histogram the IQ data in the chosen basis- #
        i_data = self.data[f'I_{basis}']
        q_data = self.data[f'Q_{basis}']
        if rng is None:
            abs_axis_displacement = np.abs(np.concatenate([i_data, q_data]))
            rng_val = (1.05 * abs_axis_displacement).max()
            rng = [(-rng_val, rng_val), (-rng_val, rng_val)]
        hiq, i, q = np.histogram2d(i_data, q_data, bins=bins, range=rng)

        # - generate the Gaussian Mixture - #
        gmm = getattr(self, f'IQ_{basis}_gmm')
        i_means, q_means = gmm.means_.transpose()
        covs = gmm.covariances_
        a0, b0, c0 = covs[0, 0, 0] ** (1 / 2), covs[0, 1, 0], covs[0, 1, 1] ** (1 / 2)
        a1, b1, c1 = covs[1, 0, 0] ** (1 / 2), covs[1, 1, 0], covs[1, 1, 1] ** (1 / 2)
        pdf0 = gaussian_2d(i, q, i_means[0], q_means[0], a0, c0, b0)
        pdf1 = gaussian_2d(i, q, i_means[1], q_means[1], a1, c1, b1)

        # - plot the IQ and Gaussian mixture data - #
        axs[0].pcolormesh(i, q, hiq.transpose(), cmap=cmap)
        axs[1].pcolormesh(i, q, pdf0 + pdf1, cmap=cmap)

        # - make axes lines, mixture centers - #
        axs[0].scatter(i_means, q_means, marker='o', facecolors='none', edgecolors='red', s=50)
        axs[1].scatter(i_means, q_means, marker='o', facecolors='none', edgecolors='red', s=50)
        for ax in axs:
            ax.hlines(0, rng[0][0], rng[0][1])
            ax.vlines(0, rng[1][0], rng[1][1])

        return fig, axs

    def plot_rotation(self, cmap='viridis', bases=None, axes_limits=None, axs=None):
        """ Generate grid of subplots to demonstrate the rotated bases.

        :param cmap: String name of the matplotlib colormap to use.
        :param bases: List of strings indicating which bases to plot.
        :param axes_limits: Dictionary with axes limits for each basis.
        :param axs: Array of Matplotlib Axes objects on which to plot.
        :return: Matplotlib Figure and list of Axes objects generated.
        """
        # - make the figure layout - #
        if bases is None:
            bases = ['raw', 'rot', 'phase']
        fig = None
        if axs is None:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(self.get_metadata_string())
            gs = mpl.gridspec.GridSpec(
                figure=fig,
                nrows=3, ncols=2,
                hspace=0.4, wspace=0.4,
                top=0.9, bottom=0.05,
                left=0.15, right=0.95,
            )
            axs = gs.subplots()

        # - plot all bases using plot_gaussian_mixture - #
        rng = None
        for i, basis in enumerate(bases):
            if axes_limits is not None:
                rng = axes_limits[basis]
            self.plot_gaussian_mixture(basis=basis, cmap=cmap, rng=rng, axs=axs[i, :])

        return fig, axs

    def plot_time_domain_stacked(self, data_axis_label='amps', xaxis_seconds=1, plot_title=None, saveplot=True):
        """ Splits the data into chunks of xaxis_seconds and plots them stacked on top of each other
        ----------
        Args:
            data_axis_label: str
                Key in the data dictionary indicating the data to plot
            xaxis_seconds: float
                Length of time for each plot
            plot_title: str
                Title for the plot. If None, a title is generated from metadata
            saveplot: bool
                If True, saves the figure to the path saved to the class metadata
        """
        ## Figure out how many plots we need
        times_sec = self.data['times_ms'] / 1e3
        n_plots = int(np.ceil(times_sec[-1] / xaxis_seconds))
        n_points_per_plot = int(np.ceil(len(times_sec) / n_plots))

        ## Check if the data is in the dictionary
        if data_axis_label not in self.data.keys():
            print('Error: axis label not found in data dictionary')
            return None

        ## Make the plots
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            axes = [ax]  # wrap ax in a list for consistency in the loop
        else:
            fig, ax = plt.subplots(n_plots, 1, figsize=(8, 2.5*n_plots))
            axes = ax if isinstance(ax, np.ndarray) else [ax]

        for i in range(n_plots):
            axes[i].plot(times_sec[i*n_points_per_plot:(i+1)*n_points_per_plot],
                         self.data[data_axis_label][i*n_points_per_plot:(i+1)*n_points_per_plot],
                         color=self._qubit_colors[(self.metadata['device']+self.metadata['qubit'])],
                         linewidth=0.75)
            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel(data_axis_label)
            axes[i].text(0.02, 0.02, f'Basis: {data_axis_label}', transform=axes[i].transAxes,
                         fontsize=13, color='gray')
            #axes[i].set_ylim(min(self.data[data_axis_label][i*n_points_per_plot:(i+1)*n_points_per_plot])-2,
            #                max(self.data[data_axis_label][i*n_points_per_plot:(i+1)*n_points_per_plot])+2)

        if plot_title is None:
            plot_title = self.get_metadata_string()
        fig.suptitle(plot_title)
        plt.tight_layout()
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_time_domain.png'
            fig.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_filtered_time_domain(self, xaxis_seconds=1, filters_to_plot=None, plot_title=None,
                                  show_all_data=True, saveplot=True):
        """ Splits the data into chunks of xaxis_seconds, filters them, and plots them stacked on top of each other
        ----------
        Args:
            xaxis_seconds: float
                Length of time for each plot
            filters_to_plot: list
                List of filters to plot. If None, plots all filters applied
            plot_title: str
                Title for the plot. If None, a title is generated from metadata
            show_all_data: bool
                If False, only shows first plot up the time of xaxis_seconds. Rest of data is ignored
            saveplot: bool
                If True, saves the figure to the path saved to the class metadata
        """

        try:
            datacolor = self._qubit_colors[(self.metadata['device']+self.metadata['qubit'])]
        except KeyError:
            datacolor = 'steelblue'

        ## Figure out what data to plot
        if filters_to_plot is None:
            filters_to_plot = self.filt_data['filters_applied']
        if len(filters_to_plot) == 0:
            print('Error: no filters have been applied yet')
            return None

        ## Check that the filters were applied to the same data axis
        data_axes = builtins.set([self.analysis[f'{filt}_axis'] for filt in filters_to_plot])
        #--- [HM] oops looks like we overwrote the builtin 'set' function
        if len(data_axes) > 1:
            print('Warning: filters were applied to different data axes')

        ## Figure out how many plots we need
        times_sec = self.data['times_ms'] / 1e3
        n_plots = int(np.ceil(times_sec[-1] / xaxis_seconds))
        n_points_per_plot = int(np.ceil(len(times_sec) / n_plots))
        if not show_all_data:
            n_plots = 1

        ## Make the plots
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            axes = [ax]
        else:
            fig, ax = plt.subplots(n_plots, 1, figsize=(8, 3*n_plots))
            axes = ax if isinstance(ax, np.ndarray) else [ax]

        for i in range(n_plots):
            ## plot the original data
            for datnum, data_axis in enumerate(data_axes):
                #if datnum == 0: datacolor = 'lightblue'
                #elif datnum == 1: datacolor = 'lightgreen'
                axes[i].plot(times_sec[i*n_points_per_plot:(i+1)*n_points_per_plot],
                             self.data[data_axis][i*n_points_per_plot:(i+1)*n_points_per_plot],
                             color=datacolor,
                             alpha=0.5,
                             #color=self._qubit_colors[(self.metadata['device']+self.metadata['qubit'])],
                             linewidth=1, label=data_axis)
            ## plot the filtered data
            for filt in filters_to_plot:
                filt_data = self.filt_data[f'{filt}_{self.analysis[f"{filt}_axis"]}']
                filt_data_axis = self.analysis[f'{filt}_axis']
                axes[i].plot(times_sec[i*n_points_per_plot:(i+1)*n_points_per_plot],
                             filt_data[i*n_points_per_plot:(i+1)*n_points_per_plot],
                             color=datacolor,
                             label=filt+' '+filt_data_axis, linewidth=2)
            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel(self.analysis[f'{filters_to_plot[0]}_axis'])
            #axes[i].set_ylim(min(filt_data[i*n_points_per_plot:(i+1)*n_points_per_plot])-2,
            #                max(filt_data[i*n_points_per_plot:(i+1)*n_points_per_plot])+2)
            axes[i].legend(loc='upper right')

        if plot_title is None:
            plot_title = self.get_metadata_string()
        fig.suptitle(plot_title)
        plt.tight_layout()
        if saveplot:
            if not os.path.exists(self.metadata['plotpath']):
                os.makedirs(self.metadata['plotpath'])
            plot_filename = self.metadata['filename'][:-4] + '_filtered_time_domain.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_tunneling_histogram_vs_time(self, plot_title=None, saveplot=True, ax=None):
        """
        Plots a histogram of the threshold crossing events over time bins
        ----------
        Args:
            plot_title: str
                Title for the plot. If None, a title is generated from metadata
            saveplot: bool
                If True, saves the figure to the path saved to the class metadata
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='Time (s)', ylabel='Number of Tunneling Events')

        # Check if crossing event counts are already calculated
        if 'crossings_counts_binned' not in self.analysis or 'crossings_time_bins_ms' not in self.analysis:
            print('Error: No crossing events data found. Run count_events() first.')
            return
        data_axis_label = self.analysis['crossings_data_axis_label']
        bin_width_ms = self.analysis['crossings_time_bins_ms'][1] - self.analysis['crossings_time_bins_ms'][0]
        histogram_metadata = f' - {data_axis_label} - Bin width: {bin_width_ms} ms'
        if plot_title is None:
            plot_title = self.get_metadata_string() + '\n' + histogram_metadata
        ax.set_title(plot_title)

        # Extract the previously calculated event counts and time bins
        event_counts = self.analysis['crossings_counts_binned']
        time_bins = self.analysis['crossings_time_bins_ms'] / 1000  # Convert to seconds for the x-axis

        # Plot histogram
        ax.bar(time_bins, event_counts, width=(bin_width_ms / 1000), color='cornflowerblue', edgecolor='black', align='edge')
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_tunneling_histogram.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_IQ_time_domain(self, Ival_data_axis_label='I_raw', Qval_data_axis_label='Q_raw',
                            n_plots=2, plot_title=None, saveplot=True):
        """ Splits the data into chunks of xaxis_seconds then plots it in IQ space
        ----------
        Args:
            Ival_data_axis_label: str
                Key in the data dictionary indicating the I data to plot
            Qval_data_axis_label: str
                Key in the data dictionary indicating the Q data to plot
            plot_title: str
                Title for the plot. If None, a title is generated from metadata
            saveplot: bool
                If True, saves the figure to the path saved to the class metadata
        """
        ## Figure out how many plots we need
        times_sec = self.data['times_ms'] / 1e3
        n_points_per_plot = int(np.ceil(len(times_sec) / n_plots))

        ## Find the I and Q data
        if Ival_data_axis_label in self.data.keys():
            I_data = self.data[Ival_data_axis_label]
        elif Ival_data_axis_label in self.filt_data.keys():
            I_data = self.filt_data[Ival_data_axis_label]
        else:
            print('Error: I axis label not found in data dictionary'); return None
        if Qval_data_axis_label in self.data.keys():
            Q_data = self.data[Qval_data_axis_label]
        elif Qval_data_axis_label in self.filt_data.keys():
            Q_data = self.filt_data[Qval_data_axis_label]
        else:
            print('Error: Q axis label not found in data dictionary'); return None

        ## Make the plots
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            axes = [ax]  ## wrap ax in a list for consistency in the loop
        else:
            fig, ax = plt.subplots(n_plots, 1, figsize=(4.5, 4*n_plots))
            axes = ax if isinstance(ax, np.ndarray) else [ax]

        cm = plt.get_cmap('YlGnBu')
        for i in range(n_plots):
            start_idx = i * n_points_per_plot
            end_idx = min((i + 1) * n_points_per_plot, len(times_sec))

            ## Select data for the current plot
            Ivals = I_data[start_idx:end_idx]
            Qvals = Q_data[start_idx:end_idx]
            times_chunk = self.data['times_ms'][start_idx:end_idx]

            ## Normalize based on the time range for the current chunk
            norm = plt.Normalize(vmin=times_chunk.min(), vmax=times_chunk.max())

            ## Scatter plot for this chunk of data
            scatter = axes[i].scatter(Ivals, Qvals, c=times_chunk, cmap=cm, norm=norm, marker='.', s=10)
            axes[i].set_xlabel(Ival_data_axis_label)
            axes[i].set_ylabel(Qval_data_axis_label)
            axes[i].text(0.02, 0.02, f'Axes: {Ival_data_axis_label} & {Qval_data_axis_label}',
                         transform=axes[i].transAxes, fontsize=13, color='gray')

            ## Create a color bar for each individual plot
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label('Time [ms]')
            cbar.set_ticks([times_chunk.min(), times_chunk.max()])
            cbar.ax.set_yticklabels([f'{times_chunk.min():.1f}', f'{times_chunk.max():.1f}'])

        if plot_title is None:
            plot_title = self.get_metadata_string()
        fig.suptitle(plot_title)
        plt.tight_layout()
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_time_domain.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

    def plot_tunneling_histogram_vs_nevents(self, plot_title=None, saveplot=True, ax=None):
        """
        Makes a histogram where the x-axis is number of events,
        and y axis is number of bins containing this number of events.

        Args:
            plot_title: str
                Title for the plot. If None, a title is generated from metadata.
            saveplot: bool
                If True, saves the figure to the path saved to the class metadata.
            ax: Matplotlib Axes object, optional.
                If provided, this object will be plotted on rather than creating a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='Number of events in the bin', ylabel='Number of bins')

        ## Check if crossing event counts are already calculated
        if 'crossings_counts_binned' not in self.analysis or 'crossings_time_bins_ms' not in self.analysis:
            print('Error: No crossing events data found. Run count_events() first.')
            return

        ## Pull data and tally
        data_axis_label = self.analysis['crossings_data_axis_label']
        event_counts = self.analysis['crossings_counts_binned']
        n_events = np.arange(0, np.max(event_counts)+1)
        n_events_counts = np.zeros(len(n_events))
        bin_width_ms = self.analysis['crossings_time_bins_ms'][1] - self.analysis['crossings_time_bins_ms'][0]
        for i in range(len(n_events)):
            n_events_counts[i] = np.sum(event_counts == i)

        ## Plot histogram
        ax.bar(n_events, n_events_counts, color='mediumpurple', edgecolor='black', align='center', width=0.95)
        ax.set_xticks(n_events)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if plot_title is None:
            plot_title = self.get_metadata_string() + f'\n{data_axis_label} ; {bin_width_ms} ms bins'
        ax.set_title(plot_title)
        if saveplot:
            plot_filename = self.metadata['filename'][:-4] + '_tunneling_histogram_event_nums.png'
            plt.savefig(os.path.join(self.metadata['plotpath'], plot_filename))

        return fig, ax

class SimulatedParityTimestream(ParityTimestream): 
        
    def __init__(self, SNR, Gp, tel_cen, tel_sep, t_exp, ifbw, plot_fname, plotpath, dt=5e-7, hist_bins=100):
        # - Dictionaries hold metadata and data
        self.metadata = {
            'series': 'YYYYMMDD_HHMMSS',
            'SNR': SNR, # dimensionless SNR between the two states (1/sig_s)^2
            'Gamma_p': Gp, # parity switching rate, Hz
            'tel_cen': tel_cen, # generated timestream telegraph signal center
            'tel_sep': tel_sep, # ditto, separation of states. Don't confuse with the self.analysis versions, which are derived
            't_exp': t_exp, # timestream duration, s
            'bandwidth': ifbw, # sampling rate to mimic, Hz
            'dt': dt, # raw sampling rate, Hz
            'DAQ_method': 'simulated',
            'psd_done': False,
            'device': 'MC',
            'qubit': 'MC',
            'filename': plot_fname,
            'plotpath': plotpath
        }
        self.data = {}

        # - PSD and analysis results that SHOULD be written to file
        self.analysis = {}

        # - filtered data that should never be written to file
        self.filt_data = {'filters_applied': [] }
        
        # - generate a timestream - #
        self.generate_timestream() # saved to self.data['times_ms'] and ['Phase_phase']
        
    def generate_timestream(self):
        """
        From the class metadata fields SNR, Gamma_p, t_exp, and dt
        generate a telegraph signal timestream with poisson rate Gp,
        switching between normal distributions with sigma = 1/sqrt(SNR) and means +/- 1/2.
        This mimicks "Phase_phase" data for actual measurements.
        """
        
        times = np.arange(0,self.metadata['t_exp'],self.metadata['dt'])

        # generate telegraph signal
        switch_times = np.where(poisson.rvs(self.metadata['Gamma_p']*self.metadata['dt'], size=len(times)))[0]
        telegraph_pulses = np.zeros_like(times)
        telegraph_pulses[switch_times[::2]] += -1
        telegraph_pulses[switch_times[1::2]] += 1
        telegraph_sig = np.cumsum(telegraph_pulses) + 1/2

        # generate a normal distribution, sample from it or a shifted version
        rand = np.random.default_rng()
        norm_dist = rand.normal(loc=self.metadata['tel_cen']+self.metadata['tel_sep']/2, scale=1/np.sqrt(self.metadata['SNR']), size=len(times))
        norm_dist[np.where(telegraph_sig < 0)] -= self.metadata['tel_sep']
        
        # average down samples to achieve the desired sampling rate
        t_int = 1 / self.metadata['bandwidth']
        ind_int = int(t_int / self.metadata['dt'])
        N_int = int(len(norm_dist) / ind_int)

        resampled_norm_dist = np.mean(norm_dist[:N_int*ind_int].reshape(-1,ind_int), axis=1)
        resampled_times = np.mean(times[:N_int*ind_int].reshape(-1,ind_int), axis=1)
        
        # this is supposed to be a phase signal, so wrap to +/- pi
        resampled_norm_dist[resampled_norm_dist > np.pi] -= np.pi
        resampled_norm_dist[resampled_norm_dist < -np.pi] += np.pi

        self.data['times_ms'] = 1e3*resampled_times
        self.data['Phase_phase'] = resampled_norm_dist