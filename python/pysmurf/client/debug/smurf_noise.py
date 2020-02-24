#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf debug module - SmurfNoiseMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/debug/smurf_noise.py
# Created    : 2018-09-17
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from scipy import optimize
from pysmurf.client.base import SmurfBase
import os
import time
from pysmurf.client.util import tools
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class SmurfNoiseMixin(SmurfBase):

    def take_noise_psd(self, meas_time,
        channel=None, nperseg=2**12,
        detrend='constant', fs=None, low_freq=np.array([.1, 1.]),
        high_freq=np.array([1., 10.]), make_channel_plot=True,
        make_summary_plot=True, plotname_append='',
        save_data=False, show_plot=False,
        grid_on=False, datafile=None, downsample_factor=None,
        write_log=True):

        """
        Takes a timestream of noise and calculates its PSD. It also
        attempts to fit a white noise and 1/f component to the data.
        It takes into account the sampling frequency and the downsampling
        filter and downsampler.

        Args:
        -----
        meas_time (float): The amount of time to observe in seconds.
        Opt Args:
        ---------
        channel (int array): The channels to plot. Note that this script always
            takes data on all the channels. This only sets the ones to plot.
            If None, plots all channels that are on. Default is None.
        nperseg (int): The number of elements per segment in the PSD. Default
            2**12.
        detrend (str): Extends the scipy.signal.welch detrend. Default is
            'constant'
        fs (float): Sample frequency. If None, reads it in. Default is None.
        low_freq (float array):
        high_freq (float array):
        make_channel_plot (bool): Whether to make the individual channel
            plots. Default is True.
        make_summary_plot (bool): Whether to make the summary plots. Default
            is True.
        save_data (bool): Whether to save the band averaged data as a text file.
            Default is False.
        plotname_append (string): Appended to the default plot filename. Default ''.
        show_plot (bool): Show the plot on the screen. Default False.
        datefile (str): if data has already been taken, can point to a file to
            bypass data taking and just analyze.
        downsample_factor (int): The datarate is the flux ramp rate divided by
            the downsample_factor.
        write_log (bool) : Whether to write to the log file (or the screen
            if the logfile is not defined). Default is True.
        """
        if datafile == None:
            datafile = self.take_stream_data(meas_time,
                                             downsample_factor=downsample_factor,
                                             write_log=write_log)
        else:
            self.log(f'Reading data from {datafile}')

        basename, _ = os.path.splitext(os.path.basename(datafile))

        # Get downsample filter params
        filter_b = self.get_filter_b()
        filter_a = self.get_filter_a()

        timestamp, phase, mask = self.read_stream_data(datafile)
        bands, channels = np.where(mask!=-1)

        phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

        flux_ramp_freq = self.get_flux_ramp_freq() * 1.0E3

        if fs is None:
            if downsample_factor is None:
                downsample_factor = self.get_downsample_factor()
            # flux ramp rate returns in kHz
            fs = flux_ramp_freq/downsample_factor

        # Generate downsample transfer function - downsampling is at
        # flux ramp freq
        downsample_freq, downsample_transfer = signal.freqz(filter_b,
            filter_a, worN=np.arange(.01, fs/2, .01), fs=flux_ramp_freq)
        downsample_transfer = np.abs(downsample_transfer)**2

        if write_log:
            self.log(f'Plotting {bands}, {channels}', self.LOG_USER)

        n_channel = len(channels)

        if make_summary_plot or make_channel_plot:
            plt.rcParams["patch.force_edgecolor"] = True

        noise_floors = np.full((len(low_freq), n_channel), np.nan)
        f_knees = np.full(n_channel,np.nan)
        res_freqs = np.full(n_channel,np.nan)

        wl_list = []
        f_knee_list = []
        n_list = []

        plt.ion()
        if not show_plot:
            plt.ioff()
        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]

            # Calculate PSD
            f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)

            good_fit = False
            try:
                # Fit the PSD
                popt, pcov, f_fit, Pxx_fit = self.analyze_psd(f, Pxx,
                    fs=fs, flux_ramp_freq=flux_ramp_freq)
                wl, n, f_knee = popt
                if f_knee != 0.:
                    wl_list.append(wl)
                    f_knee_list.append(f_knee)
                    f_knees[c]=f_knee
                    n_list.append(n)
                    good_fit = True
                if write_log:
                    self.log(f'{c+1}. b{b}ch{ch:03}:' +
                         ' white-noise level = {:.2f}'.format(wl) +
                         ' pA/rtHz, n = {:.2f}'.format(n) +
                         ', f_knee = {:.2f} Hz'.format(f_knee))
            except Exception as e:
                if write_log:
                    self.log(f'{c+1} b{b}ch{ch:03}: '+
                             'bad fit to noise model')
                    self.log(e)

            # Calculate noise in various frequency bins
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                idx = np.logical_and(f>l, f<h)
                noise_floors[i, c] = np.mean(Pxx[idx])

            if make_channel_plot:
                fig, ax = plt.subplots(2, figsize=(8,6))

                sampleNums = np.arange(len(phase[ch_idx]))
                t_array = sampleNums/fs

                # Plot the data
                ax[0].plot(t_array,phase[ch_idx] - np.mean(phase[ch_idx]))
                ax[0].set_xlabel('Time [s]')
                ax[0].set_ylabel('Phase [pA]')

                if grid_on:
                    ax[0].grid()

                ax[1].loglog(f, Pxx)
                ylim = ax[1].get_ylim()

                # Plot the fit
                if good_fit:
                    ax[1].plot(f_fit, Pxx_fit, linestyle='--', label=f'n={n:3.2f}')

                    # plot f_knee
                    ax[1].plot(f_knee, 2.*wl, linestyle='none', marker='o',
                        label=r'$f_\mathrm{knee} = ' + f'{f_knee:0.2f},' +
                        r'\mathrm{Hz}$')
                    ax[1].plot(f_fit,wl + np.zeros(len(f_fit)), linestyle=':',
                        label=r'$\mathrm{wl} = $'+ f'{wl:0.2f},' +
                        r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$')
                    ax[1].plot(downsample_freq, wl*downsample_transfer,
                               color='k', linestyle='dashdot',
                               alpha=.5, label='Lowpass')
                    ax[1].legend(loc='best')
                    ax[1].set_ylim(ylim)

                ax[1].set_xlabel('Frequency [Hz]')
                ax[1].set_xlim(f[1],f[-1])
                ax[1].set_ylabel('Amp [pA/rtHz]')

                if grid_on:
                    ax[1].grid()

                if write_log:
                    self.log(noise_floors[-1, c])

                res_freq = self.channel_to_freq(b, ch)
                res_freqs[c]=res_freq

                ax[0].set_title(f'Band {b} Ch {ch:03} - {res_freq:.2f} MHz')

                fig.tight_layout()

                plot_name = basename + \
                            f'_noise_timestream_b{b}_ch{ch:03}{plotname_append}.png'
                fig.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight')

                # Close the individual channel plots - otherwise too many
                # plots are brought to screen
                plt.close(fig)

        if save_data:
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                save_name = basename+f'_{l:3.2f}_{h:3.2f}.txt'
                outfn = os.path.join(self.plot_dir, save_name)

                np.savetxt(outfn, np.c_[res_freqs,noise_floors[i],f_knees])
                # Publish the data
                self.pub.register_file(outfn, 'noise_timestream', format='txt')

        if make_summary_plot:
            bins = np.arange(0,351,20)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(10,6))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, '{:3.2f}'.format(l) + '-' +
                    '{:3.2f} Hz'.format(h),
                    transform=ax.transAxes, fontsize=10)
                ax.set_xlabel(r'Mean noise [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')

                plot_name = basename + \

                    '{}_{}_noise_hist{}.png'.format(l, h, plotname_append)
                plt.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight')
                if show_plot:
                    plt.show()
                else:
                    plt.close()

            if len(wl_list) > 0:
                wl_median = np.median(wl_list)
                n_median = np.median(n_list)
                f_knee_median = np.median(f_knee_list)

                n_fit = len(wl_list)
                n_attempt = len(channels)

                fig,ax = plt.subplots(1,3,figsize=(10,6))
                fig.suptitle('{}: band {} noise parameters'.format(basename, b) +
                    ' ({} fit of {} attempted)'.format(n_fit, n_attempt))
                ax[0].hist(wl_list,bins=np.logspace(np.floor(np.log10(np.min(wl_list))),
                        np.ceil(np.log10(np.max(wl_list))), 10))

                ax[0].set_xlabel('White-noise level (pA/rtHz)')
                ax[0].set_xscale('log')
                ax[0].set_title(f'median = {wl_median:.3e} pA/rtHz')
                ax[1].hist(n_list)
                ax[1].set_xlabel('Noise index')
                ax[1].set_title(f'median = {n_median:.3e}')
                ax[2].hist(f_knee_list,
                    bins=np.logspace(np.floor(np.log10(np.min(f_knee_list))),
                        np.ceil(np.log10(np.max(f_knee_list))), 10))
                ax[2].set_xlabel('Knee frequency')
                ax[2].set_xscale('log')
                ax[2].set_title(f'median = {f_knee_median:.3e} Hz')
                plt.tight_layout()
                fig.subplots_adjust(top = 0.9)
                noise_params_hist_fname = basename + \
                    '_b{}_noise_params{}.png'.format(b, plotname_append)
                plt.savefig(os.path.join(self.plot_dir,
                    noise_params_hist_fname),
                    bbox_inches='tight')

                if show_plot:
                    plt.show()
                else:
                    plt.close()

        return datafile

    def turn_off_noisy_channels(self, band, noise, cutoff=150):
        """
        Args:
        -----
        band (int): The band to search
        noise (float array): The noise floors. Presumably calculated
            using take_noise_psd.
        Optional Args:
        --------------
        cutoff (float) : The value to cut at in the same units as noise.
        """
        n_channel = self.get_number_channels(band)
        for ch in np.arange(n_channel):
            if noise[ch] > cutoff:
                self.channel_off(band, ch)


    def noise_vs_tone(self, band, tones=np.arange(10,15), meas_time=30,
                      analyze=False, bias_group=None, lms_freq_hz=None,
                      fraction_full_scale=.72):
        """
        """
        timestamp = self.get_timestamp()

        # Take data
        datafiles = np.array([])
        for i, t in enumerate(tones):
            self.log('Measuring for tone power {}'.format(t))
            self.tune_band_serial(band, drive=t)
            self.tracking_setup(band, fraction_full_scale=fraction_full_scale,
                                lms_freq_hz=lms_freq_hz)
            self.check_lock(band, lms_freq_hz=lms_freq_hz)
            time.sleep(2)
            self.log(self.get_amplitude_scale_array(band))
            datafile = self.take_stream_data(meas_time)
            datafiles = np.append(datafiles, datafile)

        self.log('Saving data')
        datafile_save = os.path.join(self.output_dir, timestamp +
                                '_noise_vs_tone_datafile.txt')
        tone_save = os.path.join(self.output_dir, timestamp +
                                '_noise_vs_tone_tone.txt')

        np.savetxt(datafile_save,datafiles, fmt='%s')
        self.pub.register_file(datafile_save, 'noise_vs_tone_data', format='txt')

        np.savetxt(tone_save, tones, fmt='%i')
        self.pub.register_file(tone_save, 'noise_vs_tone_tone', format='txt')

        #self.set_amplitude_scale_array(band, x, wait_after=1)

        #self.set_amplitude_scale_array(band, start_tones)

        if analyze:
            self.analyze_noise_vs_tone(tone_save, datafile_save,
                                       band=band, bias_group=bias_group)



    def noise_vs_bias(self, band, bias_group,bias_high=1.5, bias_low=0.,
                      step_size=0.25, bias=None, high_current_mode=True,
                      overbias_voltage=9., meas_time=30., analyze=False,
                      channel=None, nperseg=2**13, detrend='constant',
                      fs=None, show_plot=False, cool_wait=30.,
                      psd_ylim=(10.,1000.),make_timestream_plot=False,
                      only_overbias_once=False):
        """
        This ramps the TES voltage from bias_high to bias_low and takes noise
        measurements. You can make it analyze the data and make plots with the
        optional argument analyze=True. Note that the analysis is a little
        slow.
        Args:
        -----
        band (int): The band to take noise vs bias data on
        bias_group (int or int array): which bias group(s) to bias/read back.
        Opt Args:
        ---------
        bias (float array): The array of bias values to step through. If None,
            uses values in defined by bias_high, bias_low, and step_size.
        bias_high (float): The bias voltage to start at
        bias_low (float): The bias votlage to end at
        step_size (float): The step in voltage.
        overbias_voltage (float): voltage to set the overbias
        meas_time (float): The amount of time to take data at each TES bias.
        analyze (bool): Whether to analyze the data
        channel (int): The channel to run analysis on. Note that data is taken
            on all channels. This only affects what is analyzed. You can always
            run the analyze script later.
        nperseg (int): The number of samples per segment in the PSD.
        detrend (str): Whether to detrend the data before taking the PSD.
            Default is to remove a constant.
        fs (float): The sample frequency.
        show_plot: Whether to show analysis plots. Defaults to False.
        only_overbias_once (bool): Whether or not to overbias right
            before each TES bias step
        """
        if bias is None:
            if step_size > 0:
                step_size *= -1
            bias = np.arange(bias_high, bias_low-np.absolute(step_size), step_size)

        self.noise_vs(band=band, bias_group=bias_group, var='bias',
                      var_range=bias, meas_time=meas_time,
                      analyze=analyze, channel=channel,
                      nperseg=nperseg, detrend=detrend, fs=fs,
                      show_plot=show_plot, psd_ylim=psd_ylim,
                      overbias_voltage=overbias_voltage,
                      cool_wait=cool_wait,high_current_mode=high_current_mode,
                      make_timestream_plot=make_timestream_plot,
                      only_overbias_once=only_overbias_once)

    def noise_vs_amplitude(self, band, amplitude_high=11, amplitude_low=9, step_size=1,
                           amplitudes=None,
                           meas_time=30., analyze=False, channel=None, nperseg=2**13,
                           detrend='constant', fs=None, show_plot = False,
                           make_timestream_plot=False,
                           psd_ylim = None):
        """
        Args:
        -----
        band (int): The band to take noise vs bias data on
        """
        if amplitudes is None:
            if step_size > 0:
                step_size *= -1
            amplitudes = np.arange(amplitude_high,
                amplitude_low-np.absolute(step_size), step_size)

        self.noise_vs(band=band,var='amplitude',var_range=amplitudes,
                 meas_time=meas_time, analyze=analyze, channel=channel,
                 nperseg=nperseg, detrend=detrend, fs=fs, show_plot=show_plot,
                 make_timestream_plot=make_timestream_plot,
                 psd_ylim=psd_ylim)

    def noise_vs(self, band, var, var_range, meas_time=30,
                 analyze=False, channel=None, nperseg=2**13,
                 detrend='constant', fs=None, show_plot=False,
                 psd_ylim=None, make_timestream_plot=False,
                 only_overbias_once=False, **kwargs):

        if fs is None:
            fs = self.fs

        # aliases
        biasaliases=['bias']
        amplitudealiases=['amplitude']

        # vs TES bias
        if var in biasaliases:
            # requirement
            assert ('bias_group' in kwargs.keys()),'Must specify bias_group.'
            # defaults
            if 'high_current_mode' not in kwargs.keys():
                kwargs['high_current_mode']=False
            if 'cool_wait' not in kwargs.keys():
                kwargs['cool_wait']=30.

        if var in amplitudealiases:
            # no parameters (yet) but need to null this until we rework the analysis
            kwargs['bias_group']=-1
            pass

        psd_dir = os.path.join(self.output_dir, 'psd')
        self.make_dir(psd_dir)

        timestamp = self.get_timestamp()
        fn_var_values = os.path.join(psd_dir, '{}_{}.txt'.format(timestamp,var))


        np.savetxt(fn_var_values,var_range)
        # Is this an accurate tag?
        self.pub.register_file(fn_var_values, 'noise_vs_{}'.format(var),
                               format='txt')

        datafiles = np.array([], dtype=str)
        xlabel_override=None
        unit_override=None
        actually_overbias=True
        for v in var_range:
            if var in biasaliases:
                self.log('Bias {}'.format(v))
                if type(kwargs['bias_group']) is int: # only received one group
                    self.overbias_tes(kwargs['bias_group'], tes_bias=v,
                                 high_current_mode=kwargs['high_current_mode'],
                                 cool_wait=kwargs['cool_wait'],
                                 overbias_voltage=kwargs['overbias_voltage'],
                                 actually_overbias=actually_overbias)
                else:
                    self.overbias_tes_all(kwargs['bias_group'], tes_bias=v,
                                 high_current_mode=kwargs['high_current_mode'],
                                 cool_wait=kwargs['cool_wait'],
                                 overbias_voltage=kwargs['overbias_voltage'],
                                 actually_overbias=actually_overbias)
                if only_overbias_once:
                    actually_overbias=False

            if var in amplitudealiases:
                unit_override=''
                xlabel_override='Tone amplitude [unit-less]'
                self.log('Retuning at tone amplitude {}'.format(v))
                self.set_amplitude_scale_array(band,
                    np.array(self.get_amplitude_scale_array(band)*v/
                        np.max(self.get_amplitude_scale_array(band)),dtype=int))
                self.run_serial_gradient_descent(band)
                self.run_serial_eta_scan(band)
                self.tracking_setup(band,lms_freq_hz=self.lms_freq_hz[band],
                    save_plot=True, make_plot=True, channel=self.which_on(band),
                    show_plot=False)

            self.log('Taking data')
            datafile = self.take_stream_data(meas_time)
            datafiles = np.append(datafiles, datafile)
            self.log('datafile {}'.format(datafile))

        self.log(f'Done with noise vs {var}')

        fn_datafiles = os.path.join(psd_dir,
            '{}_datafiles.txt'.format(timestamp))

        np.savetxt(fn_datafiles,datafiles, fmt='%s')
        self.pub.register_file(fn_datafiles, 'datafiles', format='txt')

        self.log('Saving variables values to {}.'.format(fn_var_values))
        self.log('Saving data filenames to {}.'.format(fn_datafiles))

        if analyze:
            self.analyze_noise_vs_bias(var_range, datafiles,  channel=channel,
                                       band=band,
                                       bias_group = kwargs['bias_group'],
                                       nperseg=nperseg, detrend=detrend, fs=fs,
                                       save_plot=True, show_plot=show_plot,
                                       data_timestamp=timestamp ,psd_ylim=psd_ylim,
                                       make_timestream_plot=make_timestream_plot,
                                       xlabel_override=xlabel_override,
                                       unit_override=unit_override)

    def get_datafiles_from_file(self,fn_datafiles):
        '''
        For, e.g., noise_vs_bias, the list of datafiles is recorded in a txt file.
        This function simply extracts those filenames and returns them as a list.
        Args:
        -----
        fn_datafiles (str): full path to txt containing names of data files
        Returns: datafiles (list): strings of data-file names.
        '''
        datafiles = []
        f_datafiles = open(fn_datafiles,'r')
        for line in f_datafiles:
            datafiles.append(line.split()[0])
        return datafiles

    def get_biases_from_file(self,fn_biases,dtype=float):
        '''
        For, e.g., noise_vs_bias, the list of commanded bias voltages is
        recorded in a txt file. This function simply extracts those values and
        returns them as a list.
        Args:
        -----
        fn_biases (str): full path to txt containing list of bias voltages
        Returns biases (list): floats of commanded bias voltages
        '''
        biases = []
        f_biases = open(fn_biases,'r')
        for line in f_biases:
            bias_str = line.split()[0]
            if dtype == float:
                bias = float(bias_str)
            elif dtype == int:
                bias = int(bias_str)
            biases.append(bias)
        return biases

    def get_iv_data(self,iv_data_filename,band,high_current_mode=False):
        '''
        Takes IV data and extracts responsivities as a function of commanded
        bias voltage.
        Parameters
        ----------
        iv_data_filename (str): filename of output of IV analysis
        band (int): band from which to extract responsivities
        high_current_mode (bool): whether or not to return the IV bias
                                  voltages so that they look like the IV was
                                  taken in high-current mode

        Returns
        -------
        iv_band_data (dict): dictionary with IV information for band
        '''
        self.log('Extracting IV data from {}'.format(iv_data_filename))
        iv_data = np.load(iv_data_filename).item()
        iv_band_data = iv_data[band]
        iv_high_current_mode = iv_data['high_current_mode']
        for ch in iv_band_data:
            v_bias = iv_band_data[ch]['v_bias']
            if iv_high_current_mode and not high_current_mode:
                iv_band_data[ch]['v_bias'] = v_bias*self.high_low_current_ratio
            elif not iv_high_current_mode and high_current_mode:
                iv_band_data[ch]['v_bias'] = v_bias/self.high_low_current_ratio
        return iv_band_data

    def get_si_data(self,iv_band_data,ch):
        return iv_band_data[ch]['v_bias'], iv_band_data[ch]['si']

    def NEI_to_NEP(self,iv_band_data,ch,v_bias):
        '''
        Takes NEI in pA/rtHz and converts to NEP in aW/rtHz.
        Parameters
        ----------
        si_dict (dict): dictionary indexed by channel number; each entry is
                        an array of responsivities in uV^-1.
        iv_bias_array (array): array of commanded bias voltages from the IV
                               curve from which si was estimated. The length
                               of iv_bias_array is one greater than that of
                               each element of si_dict.
        v_bias (float): commanded bias voltage at which to estimate NEP
        Pxx (array): NEI in pA/rtHz
        Returns
        -------
        1/si (float): noise-equivalent power in aW/rtHz
        '''
        v_bias_array,si_array = self.get_si_data(iv_band_data,ch)
        si = np.interp(v_bias,v_bias_array[:-1],si_array)
        return 1./np.absolute(si)

    def analyze_noise_vs_bias(self, bias, datafile, channel=None, band=None,
        nperseg=2**13, detrend='constant', fs=None, save_plot=True,
        show_plot=False, make_timestream_plot=False, data_timestamp=None,
        psd_ylim=(10.,1000.), bias_group=None, smooth_len=15,
        show_legend=True, freq_range_summary=None, R_sh=None,
        high_current_mode=True, iv_data_filename=None, NEP_ylim=(10.,1000.),
        f_center_GHz=150.,bw_GHz=32., xlabel_override=None,
        unit_override=None):
        """
        Analysis script associated with noise_vs_bias.
        Args:
        -----
        bias (float array): The bias in voltage. Can also pass an absolute
            path to a txt containing the bias points.
        datafile (str array): The paths to the datafiles. Must be same length
            as bias array. Can also pass an absolute path to a txt containing
            the names of the datafiles.
        Opt Args:
        ---------
        channel (int array): The channels to analyze.
        band (int): The band where the data is taken.
        nperseg (int): Passed to scipy.signal.welch. Number of elements per
            segment of the PSD.
        detrend (str): Passed to scipy.signal.welch.
        fs (float): Passed to scipy.signal.welch. The sample rate.
        save_plot (bool): Whether to save the plot. Default is True.
        show_plot (bool): Whether to how the plot. Default is False.
        data_timestamp (str): The string used as a save name. Default is None.
        bias_group (int or int array): which bias groups were used. Default is None.
        smooth_len (int): length of window over which to smooth PSDs for plotting
        freq_range_summary (tup): frequencies between which to take mean noise
            for summary plot of noise vs. bias; if None, then plot white-noise
            level from model fit
        """
        if not show_plot:
            plt.ioff()

        if unit_override is None:
            unit='V'
        else:
            unit=unit_override

        n_channel = self.get_number_channels(band)
        if band is None and channel is None:
            channel = np.arange(n_channel)
        elif band is not None and channel is None:
            channel = self.which_on(band)

        if fs is None:
            #self.log('No flux ramp freq given. Loading current flux ramp' +
            #    'frequency', self.LOG_USER)
            #fs = self.get_flux_ramp_freq()*1.0E3
            fs = self.fs

        if R_sh is None:
            R_sh = self.R_sh

        if isinstance(bias,str):
            self.log(f'Biases being read from {bias}')
            bias = self.get_biases_from_file(bias)

        if isinstance(datafile,str):
            self.log(f'Noise data files being read from {datafile}')
            datafile = self.get_datafiles_from_file(datafile)

        # If an analyzed IV datafile is given, estimate NEP
        if iv_data_filename is not None and band is not None:
            iv_band_data = self.get_iv_data(iv_data_filename,band,
                                           high_current_mode=high_current_mode)
            self.log('IV data given. Estimating NEP. Skipping noise analysis'
                ' for channels without responsivity estimates.')
            est_NEP = True
        else:
            est_NEP = False

        mask = np.loadtxt(self.smurf_to_mce_mask_file)

        timestream_dict = {}
        # Analyze data and save
        for i, (b, d) in enumerate(zip(bias, datafile)):
            timestream_dict[b] = {}
            timestamp, phase, mask = self.read_stream_data(d)
            phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

            basename, _ = os.path.splitext(os.path.basename(d))
            dirname = os.path.dirname(d)
            psd_dir = os.path.join(dirname, 'psd')
            self.make_dir(psd_dir)

            for ch in channel:
                ch_idx = mask[band, ch]
                phase_ch = phase[ch_idx]
                timestream_dict[b][ch] = phase_ch
                f, Pxx = signal.welch(phase_ch, nperseg=nperseg,
                    fs=fs, detrend=detrend)
                Pxx = np.sqrt(Pxx)  # pA

                path = os.path.join(psd_dir,
                                    basename + '_psd_ch{:03}.txt'.format(ch))
                np.savetxt(path, np.array([f, Pxx]))
                self.pub.register_file(path, 'psd', format='txt')

            # Explicitly remove objects from memory
            del timestamp
            del phase

        # Make plot
        cm = plt.get_cmap('plasma')
        noise_est_data = []
        if est_NEP:
            NEP_est_data = []
        n_bias = len(bias)
        n_row = int(np.ceil(n_bias/2.)*3)
        h_NEI = int(n_row/3)
        w_NEI = 2
        h_timestream = 1
        w_timestream = w_NEI
        h_NEIwl = h_NEI
        h_NEPwl = h_NEIwl
        h_SI = n_row - h_NEIwl - h_NEPwl
        w_NEIwl = 1
        w_NEPwl = w_NEIwl
        w_SI = w_NEPwl
        n_col = w_NEI + w_NEIwl
        for ch in channel:
            if ch < 0:
                continue
            w_fig = 13
            if make_timestream_plot or est_NEP:
                h_fig = 19
            else:
                h_fig = 7
            fig = plt.figure(figsize = (w_fig,h_fig))
            gs = GridSpec(n_row,n_col)
            ax_NEI = fig.add_subplot(gs[:h_NEI,:w_NEI])
            ax_NEIwl = fig.add_subplot(gs[:h_NEIwl,w_NEI:w_NEI+w_NEIwl])
            if est_NEP:
                if ch not in iv_band_data:
                    self.log('Skipping channel {}: no responsivity data.'.format(ch))
                    continue
                ax_NEPwl = fig.add_subplot(gs[h_NEIwl:h_NEIwl+h_NEPwl,\
                                        w_timestream:w_timestream+w_NEPwl])
                ax_SI = fig.add_subplot(gs[h_NEIwl+h_NEPwl:h_NEIwl+h_NEPwl+h_SI,w_timestream:w_timestream+w_SI])
            if make_timestream_plot:
                axs_timestream = []
                for i in range(n_bias):
                    ax_i = fig.add_subplot(gs[h_NEI+i:h_NEI+i+1,:w_timestream])
                    axs_timestream.append(ax_i)

            noise_est_list = []
            if est_NEP:
                NEP_est_list = []
            for i, (b, d) in enumerate(zip(bias, datafile)):
                basename, _ = os.path.splitext(os.path.basename(d))
                dirname = os.path.dirname(d)

                self.log(os.path.join(psd_dir, basename +
                    '_psd_ch{:03}.txt'.format(ch)))

                f, Pxx =  np.loadtxt(os.path.join(psd_dir, basename +
                    '_psd_ch{:03}.txt'.format(ch)))
                if est_NEP:
                    NEI2NEP = self.NEI_to_NEP(iv_band_data,ch,b)
                    NEP = Pxx*NEI2NEP
                # smooth Pxx for plotting
                if smooth_len >= 3:
                    window_len = smooth_len
                    self.log(f'Smoothing PSDs for plotting with window of length {window_len}')
                    s = np.r_[Pxx[window_len-1:0:-1], Pxx, Pxx[-2:-window_len-1:-1]]
                    w = np.hanning(window_len)
                    Pxx_smooth_ext = np.convolve(w/w.sum(), s, mode='valid')
                    ndx_add = window_len % 2
                    Pxx_smooth = Pxx_smooth_ext[(window_len//2)-1+ndx_add:-(window_len//2)]
                else:
                    self.log('No smoothing of PSDs for plotting.')
                    Pxx_smooth = Pxx

                color = cm(float(i)/len(bias))

                label_bias = '{:.2f} {}'.format(b,unit)
                ax_NEI.plot(f, Pxx_smooth, color=color, label=label_bias)
                ax_NEI.set_xlim(min(f[1:]),max(f[1:]))
                ax_NEI.set_ylim(psd_ylim)

                if make_timestream_plot:
                    ax_i = axs_timestream[i]
                    ts_i = timestream_dict[b][ch]
                    ts_i -= np.mean(ts_i) # subtract offset
                    t_i = np.arange(len(ts_i))/fs
                    ax_i.plot(t_i,ts_i,color=color,label=label_bias)
                    ax_i.legend(loc='upper right')
                    ax_i.grid()
                    if i == n_bias - 1:
                        ax_i.set_xlabel('Time [s]')
                    else:
                        ax_i.set_xticklabels([])
                    ax_i.set_xlim(min(t_i),max(t_i))
                    ax_i.set_ylabel('Phase [pA]')

                # fit to noise model; catch error if fit is bad
                popt, pcov, f_fit, Pxx_fit = self.analyze_psd(f,Pxx)
                wl, n, f_knee = popt
                self.log('ch. {}, bias = {:.2f}'.format(ch,b) +
                         ', white-noise level = {:.2f}'.format(wl) +
                         ' pA/rtHz, n = {:.2f}'.format(n) +
                         ', f_knee = {:.2f} Hz'.format(f_knee))

                # get noise estimate to summarize PSD for given bias
                if freq_range_summary is not None:
                    freq_min,freq_max = freq_range_summary
                    idxs_est = np.logical_and(f>=freq_min,f<=freq_max)
                    noise_est = np.mean(Pxx[idxs_est])
                    self.log('ch. {}, bias = {:.2f}'.format(ch,b) +
                             ', mean current noise between ' +
                             '{:.3e} and {:.3e} Hz = {:.2f} pA/rtHz'.format(freq_min,freq_max,noise_est))
                else:
                    noise_est = wl
                noise_est_list.append(noise_est)

                if est_NEP:
                    self.log('abs(responsivity) = {:.2f} uV^-1'.format(1./NEI2NEP))
                    NEP_est = noise_est*NEI2NEP
                    self.log('power noise = {:.2f} aW/rtHz'.format(NEP_est))
                    NEP_est_list.append(NEP_est)

                ax_NEI.plot(f_fit, Pxx_fit, color=color, linestyle='--')
                ax_NEI.plot(f, wl + np.zeros(len(f)), color=color,
                        linestyle=':')
                ax_NEI.plot(f_knee, 2.*wl, marker='o', linestyle='none',
                        color=color)

                ax_NEIwl.plot(b,noise_est,color=color,marker='s',
                    linestyle='none')
                if est_NEP:
                    ax_NEPwl.plot(b,NEP_est,color=color,marker='s',
                        linestyle='none')
                    iv_bias,si = self.get_si_data(iv_band_data,ch)
                    iv_bias = iv_bias[:-1]
                    if i == 0:
                        ax_SI.plot(iv_bias,si)
                        v_tes = iv_band_data[ch]['v_tes'][:-1]
                        si_etf = -1./v_tes
                        ax_SI.plot(iv_bias,si_etf,linestyle = '--',
                                   label=r'$-1/V_\mathrm{TES}$')
                        trans_idxs = iv_band_data[ch]['trans idxs']
                        sc_idx = trans_idxs[0]
                        nb_idx = trans_idxs[1]
                        R = iv_band_data[ch]['R']
                        R_n = iv_band_data[ch]['R_n']
                        R_frac_min = R[sc_idx]/R_n
                        R_frac_max = R[nb_idx]/R_n
                        for ax in [ax_NEIwl,ax_NEPwl,ax_SI]:
                            if ax == ax_SI:
                                label_Rfrac = f'{R_frac_min:.2f}-{R_frac_max:.2f}' + \
                                    r'$R_\mathrm{N}$'
                            else:
                                label_Rfrac = None
                            ax.axvspan(iv_bias[sc_idx],iv_bias[nb_idx],
                                       alpha=.15,label=label_Rfrac)
                    ax_SI.plot(b, np.interp(b, iv_bias, si), color=color,
                               marker='s', linestyle='none')

            ax_NEI.set_xlabel(r'Freq [Hz]')
            ax_NEI.set_ylabel(r'NEI [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')
            ax_NEI.set_xscale('log')
            ax_NEI.set_yscale('log')
            if show_legend:
                ax_NEI.legend(loc = 'upper right')
            res_freq = self.channel_to_freq(band, ch)

            xrange_bias = max(bias) - min(bias)
            xbuffer_bias = xrange_bias/20.
            xlim_bias = (min(bias)-xbuffer_bias,max(bias)+xbuffer_bias)
            ax_NEIwl.set_xlim(xlim_bias)

            if xlabel_override is None:
                xlabel_bias = r'Commanded bias voltage [V]'
            else:
                xlabel_bias=xlabel_override
            if est_NEP:
                ax_SI.set_xlim(xlim_bias)
                ax_NEPwl.set_xlim(xlim_bias)
                ax_SI.set_xlabel(xlabel_bias)
                ax_NEIwl.set_xticklabels([])
                ax_NEPwl.set_xticklabels([])
            else:
                ax_NEIwl.set_xlabel(xlabel_bias)

            if freq_range_summary is not None:
                ylabel_summary = f'mean noise {freq_min:.2f}-{freq_max:.2f} Hz'
            else:
                ylabel_summary = 'white-noise level'
            ax_NEIwl.set_ylabel(f'NEI {ylabel_summary} ' +
                r'[$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')

            bottom = max(0.95*min(noise_est_list),0.)
            top_desired = 1.05*max(noise_est_list)
            if psd_ylim is not None:
                top = min(psd_ylim[1],top_desired)
            else:
                top = top_desired
            ax_NEIwl.set_ylim(bottom=bottom, top=top)
            ax_NEIwl.grid()

            if est_NEP:
                ax_NEPwl.set_ylabel(f'NEP {ylabel_summary} ' +
                    r'[$\mathrm{aW}/\sqrt{\mathrm{Hz}}$]')
                ax_SI.set_ylabel(r'Estimated responsivity with $\beta = 0$'+
                    r'[$\mu\mathrm{V}^{-1}$]')

                bottom_NEP = 0.95*min(NEP_est_list)
                top_NEP_desired = 1.05*max(NEP_est_list)
                if NEP_ylim is not None:
                    top_NEP = min(NEP_ylim[1],top_NEP_desired)
                else:
                    top_NEP = top_NEP_desired
                ax_NEPwl.set_ylim(bottom=bottom_NEP,top=top_NEP)

                ax_NEPwl.set_yscale('log')
                v_tes_target = iv_band_data[ch]['v_tes_target']
                ax_SI.set_ylim(-2./v_tes_target,0.5/v_tes_target)

                ax_NEPwl.grid()
                ax_SI.grid()

                ax_NET = ax_NEPwl.twinx()

                # NEP to NET conversion model
                def NEPtoNET(NEP):
                    return (1e-18/1e-6)*(1./np.sqrt(2.))*NEP/tools.dPdT_singleMode(f_center_GHz*1e9,bw_GHz*1e9,2.7)

                bottom_NET = NEPtoNET(bottom_NEP)
                top_NET = NEPtoNET(top_NEP)

                # labels and limits
                ax_NET.set_ylim(bottom=bottom_NET,top=top_NET)
                ax_NET.set_yscale('log')
                ax_NET.set_ylabel(r'NET with opt. eff. = $100\%$ [$\mu\mathrm{K} \sqrt{\mathrm{s}}$]')
                ax_SI.legend(loc='best')

            if type(bias_group) is not int: # ie if there were more than one
                fig_title_string = ''
                file_name_string = ''
                for i in range(len(bias_group)):
                    g = bias_group[i]
                    fig_title_string += str(g) + ',' # I'm sorry but the satellite was down
                    file_name_string += str(g) + '_'
            else:
                fig_title_string = str(bias_group) + ','
                file_name_string = str(bias_group) + '_'

            # Title and layout
            fig.suptitle(basename +
                ' Band {}, Group {}'.format(band,fig_title_string) +
                ' Channel {:03} - {:.2f} MHz'.format(ch, res_freq))
            plt.tight_layout()

            if show_plot:
                plt.show()

            if save_plot:
                plot_name = 'noise_vs_bias_band{}_g{}ch{:03}.png'.format(band,
                    file_name_string, ch)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp()) + plot_name
                plot_fn = os.path.join(self.plot_dir, plot_name)
                self.log(f'Saving plot to {plot_fn}')
                plt.savefig(plot_fn,
                    bbox_inches='tight')
                plt.close()

            del f
            del Pxx

            noise_est_dict = {'ch':ch,'noise_est_list':noise_est_list}
            noise_est_data.append(noise_est_dict)
            if est_NEP:
                NEP_est_dict = {'ch':ch,'NEP_est_list':NEP_est_list}
                NEP_est_data.append(NEP_est_dict)

        n_analyzed = len(noise_est_data)

        # make summary histogram of noise vs. bias over all analyzed channels
        noise_est_data_bias = []
        if est_NEP:
            NEP_est_data_bias = []
        for i in range(len(bias)):
            b = bias[i]
            noise_est_bias = []
            if est_NEP:
                NEP_est_bias = []
            for j in range(len(noise_est_data)):
                noise_est_bias.append(noise_est_data[j]['noise_est_list'][i])
            if est_NEP:
                for j in range(len(NEP_est_data)):
                    NEP_est_bias.append(NEP_est_data[j]['NEP_est_list'][i])
            noise_est_data_bias.append(np.array(noise_est_bias))
            if est_NEP:
                NEP_est_data_bias.append(np.array(NEP_est_bias))

        if psd_ylim is not None:
            bin_min = np.log10(psd_ylim[0])
            bin_max = np.log10(psd_ylim[1])
        else:
            bin_min = np.floor(np.log10(np.min(noise_est_data_bias)))
            bin_max = np.ceil(np.log10(np.max(noise_est_data_bias)))

        # Make figure
        plt.figure()

        # Make bins
        bins_hist = np.logspace(bin_min,bin_max,20)
        hist_mat = np.zeros((len(bins_hist)-1,len(bias)))
        noise_est_median_list = []

        for i in range(len(bias)):
            hist_mat[:,i],_ = np.histogram(noise_est_data_bias[i],bins=bins_hist)
            noise_est_median_list.append(np.median(noise_est_data_bias[i]))
        X_hist,Y_hist = np.meshgrid(np.arange(len(bias),-1,-1),bins_hist)

        plt.pcolor(X_hist,Y_hist,hist_mat)
        cbar = plt.colorbar()

        # Labels
        cbar.set_label('Number of channels')
        plt.yscale('log')
        plt.ylabel(f'NEI {ylabel_summary}' +
            r' [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')
        plt.ylim(10**bin_min,10**bin_max)
        plt.title(basename +
            ': Band {}, Group {}, {} channels'.format(band,
                fig_title_string.strip(','),n_analyzed))
        xtick_labels = []
        for b in bias:
            xtick_labels.append('{}'.format(b))
        xtick_locs = np.arange(len(bias)-1,-1,-1) + 0.5
        plt.xticks(xtick_locs, xtick_labels)
        plt.xlabel('Commanded bias voltage [V]')

        # Plot the data
        plt.plot(xtick_locs, noise_est_median_list, linestyle='--', marker='o',
                 color='r', label='Median NEI')
        plt.legend(loc='lower center')

        if show_plot:
            plt.show()
        if save_plot:
            plot_name = 'noise_vs_bias_band{}_g{}NEI_hist.png'.format(band,file_name_string)
            if data_timestamp is not None:
                plot_name = '{}_'.format(data_timestamp) + plot_name
            else:
                plot_name = '{}_'.format(self.get_timestamp()) + plot_name
            plot_fn = os.path.join(self.plot_dir, plot_name)
            self.log(f'\nSaving NEI histogram to {plot_fn}')
            plt.savefig(plot_fn, bbox_inches='tight')
            plt.close()

        if est_NEP:
            if NEP_ylim is not None:
                bin_NEP_min = np.log10(NEP_ylim[0])
                bin_NEP_max = np.log10(NEP_ylim[1])
            else:
                bin_NEP_min = np.floor(np.log10(np.min(NEP_est_data_bias)))
                bin_NEP_max = np.ceil(np.log10(np.max(NEP_est_data_bias)))

            plt.figure()
            bins_NEP_hist = np.logspace(bin_NEP_min,bin_NEP_max,20)
            hist_NEP_mat = np.zeros((len(bins_NEP_hist)-1,len(bias)))
            NEP_est_median_list = []
            for i in range(len(bias)):
                hist_NEP_mat[:,i],_ = np.histogram(NEP_est_data_bias[i],
                    bins=bins_NEP_hist)
                NEP_est_median_list.append(np.median(NEP_est_data_bias[i]))
            X_NEP_hist,Y_NEP_hist = np.meshgrid(np.arange(len(bias),-1,-1),
                bins_NEP_hist)
            plt.pcolor(X_NEP_hist,Y_NEP_hist,hist_NEP_mat)
            cbar_NEP = plt.colorbar()
            cbar_NEP.set_label('Number of channels')
            plt.yscale('log')
            plt.ylabel(f'NEP {ylabel_summary}' +
                r' [$\mathrm{aW}/\sqrt{\mathrm{Hz}}$]')
            plt.ylim(10**bin_NEP_min,10**bin_NEP_max)
            plt.title(basename +
                ': Band {}, Group {}, {} channels'.format(band,
                    fig_title_string.strip(','), n_analyzed))
            plt.xticks(xtick_locs,xtick_labels)
            plt.xlabel('Commanded bias voltage [V]')
            plt.plot(xtick_locs,NEP_est_median_list, linestyle='--', marker='o',
                 color='r', label='Median NEP')
            plt.legend(loc='lower center')
            if show_plot:
                plt.show()
            if save_plot:
                plot_name = 'noise_vs_bias_band{}_g{}NEP_hist.png'.format(band,file_name_string)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp()) + plot_name
                plot_fn = os.path.join(self.plot_dir, plot_name)
                self.log(f'\nSaving NEP histogram to {plot_fn}')
                plt.savefig(plot_fn, bbox_inches='tight')
                plt.close()


    def analyze_psd(self, f, Pxx, fs=None, flux_ramp_freq=None,
                    p0=[100.,0.5,0.01]):
        '''
        Return model fit for a PSD.
        p0 (float array): initial guesses for model fitting: [white-noise level
        in pA/rtHz, exponent of 1/f^n component, knee frequency in Hz]
        Args:
        -----
        f (float array) : The frequency information
        Pxx (float array) : The power spectral data
        Opt Args:
        ---------
        fs (float) : Sampling frequency. If None, loads in the current
            sampling frequency.
        flux_ramp_freq (float) : The flux ramp frequency in Hz
        p0 (float array) : Initial guess for fitting PSDs
        Ret:
        ----
        popt (float array) : The fit parameters - [white_noise_level, n, f_knee]
        pcov (float array) : Covariance matrix
        f_fit (float array) : The frequency bins of the fit
        Pxx_fit (flot array) : The amplitude
        '''
        # incorporate timestream filtering
        b = self.get_filter_b()
        a = self.get_filter_a()

        if flux_ramp_freq is None:
            flux_ramp_freq = self.get_flux_ramp_freq()*1.0E3

        if fs is None:
            fs = flux_ramp_freq/self.get_downsample_factor()

        def noise_model(freq, wl, n, f_knee):
            '''
            Crude model for noise modeling.
            wl (float): white-noise level
            n (float): exponent of 1/f^n component
            f_knee (float): frequency at which white noise = 1/f^n component
            '''
            A = wl*(f_knee**n)

            # The downsample filter is at the flux ramp frequency
            w, h = signal.freqz(b, a, worN=freq, fs=flux_ramp_freq)
            tf = np.absolute(h)**2 # filter transfer function

            return (A/(freq**n) + wl)*tf

        bounds_low = [0.,0.,0.] # constrain 1/f^n to be red spectrum
        bounds_high = [np.inf,np.inf,np.inf]
        bounds = (bounds_low,bounds_high)

        try:
            popt, pcov = optimize.curve_fit(noise_model, f[1:], Pxx[1:],
                                            p0=p0,bounds=bounds)
        except Exception as e:
            wl = np.mean(Pxx[1:])
            self.log('Unable to fit noise model. ' +
                f'Reporting mean noise: {wl:.2f} pA/rtHz')

            popt = [wl, 1., 0.]
            pcov = None

        df = f[1] - f[0]
        f_fit = np.arange(f[1],f[-1] + df,df/10.)
        Pxx_fit = noise_model(f_fit,*popt)

        return popt, pcov, f_fit, Pxx_fit

    def noise_all_vs_noise_solo(self, band, meas_time=10):
        """
        Measures the noise with all the resonators on, then measures
        every channel individually.
        Args:
        -----
        band (int) : The band number
        Opt Args:
        ---------
        meas_time (float) : The measurement time per resonator in
            seconds. Default is 10.
        """
        timestamp = self.get_timestamp()

        channel = self.which_on(2)
        n_channel = len(channel)
        drive = self.freq_resp[band]['drive']

        self.log('Taking noise with all channels')

        filename = self.take_stream_data(meas_time=meas_time)

        ret = {'all': filename}

        for i, ch in enumerate(channel):
            self.log('ch {:03} - {} of {}'.format(ch, i+1, n_channel))
            self.band_off(band)
            self.flux_ramp_on()
            self.set_amplitude_scale_channel(band, ch, drive)
            self.set_feedback_enable_channel(band, ch, 1, wait_after=1)
            filename = self.take_stream_data(meas_time)
            ret[ch] = filename

        path = os.path.join(self.output_dir, timestamp + 'all_vs_solo')
        np.save(path, ret)
        self.pub.register_file(path, 'noise', format='npy')

        return ret

    def analyze_noise_all_vs_noise_solo(self, ret, fs=None, nperseg=2**10,
                                        make_channel_plot=False):
        """
        analyzes the data from noise_all_vs_noise_solo
        Args:
        -----
        ret (dict) : The returned values from noise_all_vs_noise_solo.
        """
        if fs is None:
            fs = self.fs

        keys = ret.keys()
        all_dir = ret.pop('all')

        t, d, m = self.read_stream_data(all_dir)
        d *= self.pA_per_phi0/(2*np.pi)  # convert to pA

        wl_diff = np.zeros(len(keys))

        for i, k in enumerate(ret.keys()):
            self.log('{} : {}'.format(k, ret[k]))
            tc, dc, mc = self.read_stream_data(ret[k])
            dc *= self.pA_per_phi0/(2*np.pi)
            band, channel = np.where(mc != -1)  # there should be only one

            ch_idx = m[band, channel][0]
            f, Pxx = signal.welch(d[ch_idx], fs=fs, nperseg=nperseg)
            Pxx = np.sqrt(Pxx)
            popt, pcov, f_fit, Pxx_fit = self.analyze_psd(f, Pxx)
            wl, n, f_knee = popt  # extract fit parameters

            f_solo, Pxx_solo = signal.welch(dc[0], fs=fs, nperseg=nperseg)
            Pxx_solo = np.sqrt(Pxx_solo)
            popt_solo, pcov_solo, f_fit_solo, Pxx_fit_solo = \
                self.analyze_psd(f, Pxx_solo)
            wl_solo, n_solo, f_knee_solo = popt_solo
            if make_channel_plot:
                fig, ax = plt.subplots(2)
                ax[0].plot(t-t[0], d[ch_idx]-np.median(d[ch_idx]))
                ax[0].plot(tc-tc[0], dc[0]-np.median(dc[0]))
                ax[1].semilogy(f, Pxx, alpha=.5, color='b')
                ax[1].semilogy(f, Pxx_solo, alpha=.5, color='r')
                ax[1].axhline(wl, color='b')
                ax[1].axhline(wl_solo, color='r')
                plt.show()

            wl_diff[i] = wl - wl_solo

        return wl_diff


    def NET_CMB(self, NEI, V_b, R_tes, opt_eff, f_center=150e9, bw=32e9,
        R_sh=None, high_current_mode=False):
        '''
        Converts current spectral noise density to NET in uK rt(s). Assumes NEI
        is white-noise level.
        Args
        ----
        NEI (float): current spectral density in pA/rtHz
        V_b (float): commanded bias voltage in V
        R_tes (float): resistance of TES at bias point in Ohm
        opt_eff (float): optical efficiency (in the range 0-1)
        Opt Args:
        ---------
        f_center (float): center optical frequency of detector in Hz, e.g., 150 GHz for E4c
        bw (float): effective optical bandwidth of detector in Hz, e.g., 32 GHz for E4c
        R_sh (float): shunt resistance in Ohm; defaults to stored config figure
        high_current_mode (bool): whether the bias voltage was set in high-current mode
        Ret:
        ----
        NET (float) : The noise equivalent temperature in units of uKrts
        '''
        NEI *= 1e-12 # bring NEI to SI units, i.e., A/rt(Hz)
        if high_current_mode:
            V_b *= self.high_low_current_ratio
        I_b = V_b/self.bias_line_resistance # bias current running through shunt+TES network
        if R_sh is None:
            R_sh = self.R_sh
        V_tes = I_b*R_sh*R_tes/(R_sh+R_tes) # voltage across TES
        NEP = V_tes*NEI # power spectral density
        T_CMB = 2.7
        dPdT = opt_eff*tools.dPdT_singleMode(f_center,bw,T_CMB)
        NET_SI = NEP/(dPdT*np.sqrt(2.)) # NET in SI units, i.e., K rt(s)

        return NET_SI/1e-6 # NET in uK rt(s)


    def analyze_noise_vs_tone(self, tone, datafile, channel=None, band=None,
        nperseg=2**13, detrend='constant', fs=None, save_plot=True,
        show_plot=False, make_timestream_plot=False, data_timestamp=None,
        psd_ylim=(10.,1000.), bias_group=None, smooth_len=11,
        show_legend=True, freq_range_summary=None):
        """
        Analysis script associated with noise_vs_tone.
        """

        if not show_plot:
            plt.ioff()

        n_channel = self.get_number_channels(band)
        if band is None and channel is None:
            channel = np.arange(n_channel)
        elif band is not None and channel is None:
            channel = self.which_on(band)

        if fs is None:
            fs = self.fs

        if isinstance(tone,str):
            self.log(f'Tone powers being read from {tone}')
            tone = self.get_biases_from_file(tone,dtype=int)

        if isinstance(datafile,str):
            self.log(f'Noise data files being read from {datafile}')
            datafile = self.get_datafiles_from_file(datafile)

        mask = np.loadtxt(self.smurf_to_mce_mask_file)

        # Analyze data and save
        for i, (b, d) in enumerate(zip(tone, datafile)):
            timestamp, phase, mask = self.read_stream_data(d)
            phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

            basename, _ = os.path.splitext(os.path.basename(d))
            dirname = os.path.dirname(d)
            psd_dir = os.path.join(dirname, 'psd')
            self.make_dir(psd_dir)

            for ch in channel:
                ch_idx = mask[band, ch]
                f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
                    fs=fs, detrend=detrend)
                Pxx = np.ravel(np.sqrt(Pxx))  # pA

                path = os.path.join(psd_dir,
                                    basename + '_psd_ch{:03}.txt'.format(ch))
                np.savetxt(path, np.vstack((f, Pxx)))
                self.pub.register_file(path, 'psd', format='txt')

                if make_timestream_plot:
                    fig,ax = plt.subplots(1)
                    ax.plot(phase[ch])
                    res_freq = self.channel_to_freq(band, ch)
                    ax.set_title('Channel {:03} - {:5.4f} MHz'.format(ch, res_freq))
                    ax.set_xlabel(r'Time index')
                    ax.set_ylabel(r'Phase (pA)')

                    if show_plot:
                        plt.show()
                    if save_plot:
                        plt.savefig(os.path.join(self.plot_dir, basename + \
                                    '_timestream_ch{:03}.png'.format(ch)),\
                                    bbox_inches='tight')
                        plt.close()

            # Explicitly remove objects from memory
            del timestamp
            del phase

        # Make plot
        cm = plt.get_cmap('plasma')
        fig_width = 8
        n_col = 3
        for ch in channel:
            fig = plt.figure(figsize = (fig_width,5))
            gs = GridSpec(1,n_col)
            ax0 = fig.add_subplot(gs[:2])
            ax1 = fig.add_subplot(gs[2])

            noise_est_list = []
            for i, (b, d) in enumerate(zip(tone, datafile)):
                basename, _ = os.path.splitext(os.path.basename(d))
                dirname = os.path.dirname(d)

                self.log(os.path.join(psd_dir, basename +
                    '_psd_ch{:03}.txt'.format(ch)))

                f, Pxx =  np.loadtxt(os.path.join(psd_dir, basename +
                    '_psd_ch{:03}.txt'.format(ch)))
                # smooth Pxx for plotting
                if smooth_len >= 3:
                    window_len = smooth_len
                    self.log(f'Smoothing PSDs for plotting with window of length {window_len}')
                    s = np.r_[Pxx[window_len-1:0:-1],Pxx,Pxx[-2:-window_len-1:-1]]
                    w = np.hanning(window_len)
                    Pxx_smooth_ext = np.convolve(w/w.sum(),s,mode='valid')
                    ndx_add = window_len % 2
                    Pxx_smooth = Pxx_smooth_ext[(window_len//2)-1+ndx_add:-(window_len//2)]
                else:
                    self.log('No smoothing of PSDs for plotting.')
                    Pxx_smooth = Pxx

                color = cm(float(i)/len(tone))
                ax0.plot(f, Pxx_smooth, color=color, label='{}'.format(b))
                ax0.set_xlim(min(f[1:]),max(f[1:]))
                ax0.set_ylim(psd_ylim)

                # fit to noise model; catch error if fit is bad
                popt,pcov,f_fit,Pxx_fit = self.analyze_psd(f,Pxx)
                wl,n,f_knee = popt
                self.log('ch. {}, tone power = {}'.format(ch,b) +
                         ', white-noise level = {:.2f}'.format(wl) +
                         ' pA/rtHz, n = {:.2f}'.format(n) +
                         ', f_knee = {:.2f} Hz'.format(f_knee))

                # get noise estimate to summarize PSD for given bias
                if freq_range_summary is not None:
                    freq_min,freq_max = freq_range_summary
                    noise_est = np.mean(Pxx[np.logical_and(f>=freq_min,f<=freq_max)])
                    self.log('ch. {}, tone = {}'.format(ch,b) +
                             ', mean noise between {:.3e} and {:.3e} Hz = {:.2f} pA/rtHz'.format(freq_min,freq_max,noise_est))
                else:
                    noise_est = wl
                noise_est_list.append(noise_est)

                ax0.plot(f_fit, Pxx_fit, color=color, linestyle='--')
                ax0.plot(f, wl + np.zeros(len(f)), color=color,
                        linestyle=':')
                ax0.plot(f_knee,2.*wl,marker = 'o',linestyle = 'none',
                        color=color)

                ax1.plot(b,wl,color=color,marker='s',linestyle='none')

            ax0.set_xlabel(r'Freq [Hz]')
            ax0.set_ylabel(r'PSD [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')
            ax0.set_xscale('log')
            ax0.set_yscale('log')
            if show_legend:
                ax0.legend(loc = 'upper right')
            res_freq = self.channel_to_freq(band, ch)

            ax1.set_xlabel(r'Tone-power setting')
            if freq_range_summary is not None:
                ylabel_summary = f'Mean noise {freq_min:.2f}-{freq_max:.2f Hz}'
            else:
                ylabel_summary = r'White-noise level'
            ax1.set_ylabel(f'{ylabel_summary} ' +
                r'[$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')

            bottom = max(0.95*min(noise_est_list),0.)
            top_desired = 1.05*max(noise_est_list)
            if psd_ylim is not None:
                top = min(psd_ylim[1],top_desired)
            else:
                top = top_desired
            ax1.set_ylim(bottom=bottom, top=top)
            ax1.grid()

            if type(bias_group) is not int: # ie if there were more than one
                fig_title_string = ''
                file_name_string = ''
                for i in range(len(bias_group)):
                    g = bias_group[i]
                    fig_title_string += str(g) + ',' # I'm sorry but the satellite was down
                    file_name_string += str(g) + '_'
            else:
                fig_title_string = str(bias_group) + ','
                file_name_string = str(bias_group) + '_'

            ax[0].set_title(basename +
                ' Band {}, Group {} Channel {:03} - {:.2f} MHz'.format(band,fig_title_string,ch, res_freq))
            plt.tight_layout(rect=[0.,0.03,1.,1.0])

            if show_plot:
                plt.show()

            if save_plot:
                plot_name = 'noise_vs_tone_band{}_g{}ch{:03}.png'.format(band,file_name_string,ch)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp()) + plot_name
                plot_fn = os.path.join(self.plot_dir, plot_name)
                self.log(f'Saving plot to {plot_fn}')
                plt.savefig(plot_fn,
                    bbox_inches='tight')
                plt.close()

            del f
            del Pxx



    def noise_svd(self, d, mask, mean_subtract=True):
        """
        Calculates the SVD modes of the input data.
        Only uses the data called out by the mask
        Args:
        -----
        d (float array) : The raw data
        mask (int array) : The channel mask
        Opt Args:
        ---------
        mean_subtract (bool) : Whether to mean subtract
            before taking the SVDs.
        Ret:
        ----
        u (float array) : The SVD coefficients
        s (float array) : The SVD amplitudes
        vh (float array) : The SVD modes
        """
        dat = d[mask[np.where(mask!=-1)]]
        if mean_subtract:
            dat -= np.atleast_2d(np.mean(dat, axis=1)).T

        u, s, vh = np.linalg.svd(dat, full_matrices=True)
        return u, s, vh

    def plot_svd_summary(self, u, s, save_plot=False,
        save_name=None, show_plot=False):
        """
        Requires seaborn to be installed. Plots a heatmap
        of the coefficients and the log10 of the amplitudes.
        Args:
        -----
        u (float array) : SVD coefficients from noise_svd
        s (float array) : SVD amplitudes from noise_svd
        Opt Args:
        ---------
        save_plot (bool) : Whether to save the plot
        save_name (str) : The name of the file
        show_plot (bool) : Whether to show the plot
        """
        if not show_plot:
            plt.ioff()
        else:
            plt.ion()

        fig, ax = plt.subplots(1, 2, figsize=(10,5))

        # heatmap of coefficients
        import seaborn as sns
        n_det, _ = np.shape(u)
        n_tick = 10
        tick = n_det//n_tick
        sns.heatmap(u, vmin=-1, vmax=1, cmap='RdYlBu_r',
            xticklabels=tick, yticklabels=tick, linewidth=0,
            ax=ax[0], square=False)
        ax[0].set_xlabel('Mode Num')
        ax[0].set_ylabel('Resonator')

        # Overall mode power
        ax[1].plot(np.log10(s), '.')
        ax[1].set_ylabel(r'$\log_{10}s$')
        ax[1].set_xlabel('Mode num')

        plt.tight_layout()

        # Plot saving
        if save_plot:
            if save_name is None:
                raise IOError('To save plot, save_name must be provided')
            else:
                plt.savefig(os.path.join(self.plot_dir, save_name),
                            bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_svd_modes(self, vh, n_row=4, n_col=5, figsize=(10,7.5),
        save_plot=False, save_name=None, show_plot=False, sharey=True):
        """
        Plots the first N modes where N is n_row x n_col.
        Args:
        -----
        vh (float array) : SVD modes from noise_svd
        Opt Args:
        --------
        n_row (int) : The number of rows in the figure
        n_col (int) : The number of columns in the figure
        figsize (ints) : The size of the figure
        sharey (bool) : whether the subplots share y
        save_plot (bool) : Whether to save the plot
        save_name (str) : The name of the file
        show_plot (bool) : Whether to show the plot
        """
        if not show_plot:
            plt.ioff()
        else:
            plt.ion()


        fig, ax = plt.subplots(n_row, n_col, figsize=figsize, sharex=True,
            sharey=sharey)

        n_modes = n_row * n_col

        for i in np.arange(n_modes):
            y = i // n_col
            x = i % n_col
            ax[y,x].plot(vh[i])
            ax[y,x].text(0.04, 0.91, '{}'.format(i), transform=ax[y,x].transAxes)

        plt.tight_layout()

        # Plot saving
        if save_plot:
            if save_name is None:
                raise IOError('To save plot, save_name must be provided')
            else:
                plt.savefig(os.path.join(self.plot_dir, save_name),
                            bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()


    def remove_svd(self, d, mask, u, s, vh, modes=3):
        """
        Removes the requsted SVD modes
        Args:
        -----
        d (float array) : The input data
        u (float array) : The SVD coefficients
        s (float array) : The SVD amplitudes
        mask (int array) : The channel mask
        vh (float arrry) : The SVD modes
        Opt Args:
        ---------
        modes (int or int array) : The modes to remove. If int, removes the first
            N modes. If array, uses the modes indicated in the array. Default 3.
        Ret:
        ----
        diff (float array) : The difference of the input data matrix and the
            requested SVD modes.
        """
        n_mode = np.size(s)
        n_samp, _ = np.shape(vh)

        dat = d[mask[np.where(mask!=-1)]]

        if type(modes) is int:
            modes = np.arange(modes)

        # select the modes to subtract
        diag = np.zeros((n_mode, n_samp))
        for i, dd in enumerate(s):
            if i in modes:
                diag[i,i] = dd

        # subtraction matrix
        mat = np.dot(u, np.dot(diag, vh))

        return dat - mat