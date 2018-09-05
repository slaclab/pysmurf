import numpy as np
import os
import time
from pysmurf.base import SmurfBase


class SmurfTuneMixin(SmurfBase):
    '''
    This contains all the tuning scripts
    '''


    def find_freq(self, band, subband=np.arange(13,115), drive_power=10,
        n_read=2, make_plot=False, save_plot=True):
        '''
        Finds the resonances in a band (and specified subbands)

        Args:
        -----
        band (int) : The band to search

        Optional Args:
        --------------
        subband (int) : An int array for the subbands
        drive_power (int) : The drive amplitude
        n_read (int) : The number sweeps to do per subband
        make_plot (bool) : make the plot frequency sweep. Default False.
        save_plot (bool) : save the plot. Default True.
        save_name (string) : What to name the plot. default find_freqs.png
        '''
        f, resp = full_band_ampl_sweep(band, subband, drive_power, n_read)

        timestamp = int(time.time())  # ignore fractional seconds

        # Save data
        save_name = '{}_amp_sweep_{}.txt'
        np.savetxt(os.path.join(self.output_dir, 
            save_name.format(timestamp, 'freq')), f)
        np.savetxt(os.path.join(self.output_dir, 
            save_name.format(timestamp, 'resp')), resp)

        # Place in dictionary
        self.freq_resp[band]['subband'] = subband
        self.freq_resp[band]['f'] = f
        self.freq_resp[band]['resp'] = resp
        if 'timestamp' in self.freq_resp[band]:
            self.freq_resp[band]['timestamp'] = \
                np.append(self.freq_resp[band]['timestamp'], timestamp)
        else:
            self.freq_resp[band]['timestamp'] = np.array([timestamp])

        # Find resonances
        res_freq = self.find_all_peaks(self.freq_resp[band]['f'],
            self.freq_resp[band]['resp'], subband)
        self.freq_resp[band]['resonance'] = res_freq

        # Call plotting
        if make_plot:
            plot_find_freq(self.freq_resp[band]['f'], 
                self.freq_resp[band]['resp'], save_plot=save_plot, 
                save_name=save_name.replace('.txt', '.png').format(timestamp,
                    band))

        return f, resp

    def plot_find_freq(self, f=None, resp=None, subband=None, filename=None, 
        save_plot=True, save_name='amp_sweep.png'):
        '''
        Plots the response of the frequency sweep. Must input f and resp, or
        give a path to a text file containing the data for offline plotting.

        To do:
        Add ability to use timestamp and multiple plots

        Optional Args:
        --------------
        save_plot (bool) : save the plot. Default True.
        save_name (string) : What to name the plot. default find_freqs.png
        '''
        if subband is None:
            subband = np.arange(128)
        subband = np.asarray(subband)

        if (f is None or resp is None) and filename is None:
            self.log('No input data or file given. Nothing to plot.')
            return
        else:
            if filename is not None:
                f, resp = np.load(filename)

            import matplotlib.pyplot as plt
            cm = plt.cm.get_cmap('viridis')
            fig = plt.figure(figsize=(10,4))

            for i, sb in enumerate(subband):
                color = cm(float(i)/len(subband)/2. + .5*(i%2))
                plt.plot(f[sb,:], np.abs(resp[sb,:]), '.', markersize=4, 
                    color=color)
            plt.title("findfreqs response")
            plt.xlabel("Frequency offset (MHz)")
            plt.ylabel("Normalized Amplitude")

            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, save_name),
                    bbox_inches='tight')


    def full_band_ampl_sweep(band, subband, drive, N_read):
        """sweep a full band in amplitude, for finding frequencies

        args:
        -----
            band (int) = bandNo (500MHz band)
            subband (int) = which subbands to sweep
            drive (int) = drive power (defaults to 10)
            n_read (int) = numbers of times to sweep, defaults to 2

        returns:
        --------
            freqs (list, n_freqs x 1) = frequencies swept
            resp (array, n_freqs x 2) = complex response
        """

        digitizer_freq = self.get_digitizer_frequency_mhz(band)  # in MHz
        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        band_center = self.get_band_center_mhz(band)  # in MHz

        subband_width = 2 * digitizer_freq / n_subbands

        scan_freqs = np.arange(-3, 3.1, 0.1) # take out this hardcode

        resp = np.zeros((n_subbands, np.shape(scan_freqs)[0]), dtype=complex)
        freqs = np.zeros((n_subbands, np.shape(scan_freqs)[0]))

        subband_nos, subband_centers = get_subband_centers(band)

        self.log('Working on band {:d}'.format(band), self.LOG_INFO)
        for subband in subbands:
            self.log('sweeping subband no: {}'.format(subband), self.LOG_INFO)
            r, f = fast_eta_scan(band, subband, scan_freqs, N_read, 
                drive)
            resp[subband,:] = r
            freqs[subband,:] = f
            freqs[subband,:] = scan_freqs + \
                subband_centers[subband_nos.index(subband)]
        return freqs, resp


    def peak_finder(self, x, y, threshold):
        """finds peaks in x,y data with some threshhold

        Not currently being used
        """
        in_peak = 0

        peakstruct_max = []
        peakstruct_nabove = []
        peakstruct_freq = []

        for idx in range(len(y)):
            freq = x[idx]
            amp = y[idx]

            if in_peak == 0:
                pk_max = 0
                pk_freq = 0
                pk_nabove = 0

            if amp > threshold:
                if in_peak == 0: # start a new peak
                    n_peaks = n_peaks + 1

                in_peak = 1
                pk_nabove = pk_nabove + 1

                if amp > pk_max: # keep moving until find the top
                    pk_max = amp
                    pk_freq = freq

                if idx == len(y) or y[idx + 1] < threshhold:
                    peakstruct_max.append(pk_max)
                    peakstruct_nabove.append(pk_nabove)
                    peakstruct_freq.append(pk_freq)
                    in_peak = 0
        return peakstruct_max, peakstruct_nabove, peakstruct_freq

    def find_peak(self, freq, resp, normalize=False, 
        n_samp_drop=1, threshold=.5, margin_factor=1., phase_min_cut=1, 
        phase_max_cut=1, make_plot=False, save_plot=True, save_name=None):
        """find the peaks within a given subband

        Args:
        -----
        freqs (vector): should be a single row of the broader freqs array
        response (complex vector): complex response for just this subband

        Optional Args:
        --------------
        normalize (bool) : 
        n_samp_drop (int) :
        threshold (float) :
        margin_factor (float):
        phase_min_cut (int) :
        phase_max_cut (int) :

        Returns:
        -------_
        resonances (list of floats) found in this subband
        """
        resp_input = np.copy(resp)
        if np.isnan(resp).any():
            if np.isnan(resp).all():
                self.log("Warning - All values are NAN. Skipping", 
                    self.LOG_ERROR)                
                return
            self.log("Warning - at least one NAN. Interpolating...", 
                self.LOG_ERROR)
            idx = ~np.isnan(resp)
            resp = np.interp(freq, freq[idx], resp[idx])

        # This was what was in Cyndia and Shawns code. Im deprecating this
        # for now.
        # df = freq[1] - freq[0]
        # Idat = np.real(resp)
        # Qdat = np.imag(resp)
        # phase = np.unwrap(np.arctan2(Qdat, Idat))

        # diff_phase = np.diff(phase)
        # diff_freq = np.add(freq[:-1], df / 2)  # lose an index from diff

        # if normalize==True:
        #     norm_min = min(diff_phase[nsampdrop:-nsampdrop])
        #     norm_max = max(diff_phase[nsampdrop:-nsampdrop])

        #     diff_phase = (diff_phase - norm_min) / (norm_max - norm_min)
        #
        # peakstruct_max, peakstruct_nabove, peakstruct_freq = \
        #     self.peak_finder(diff_freq, diff_phase, threshold)

        # return peakstruct_max, peakstruct_nabove, peakstruct_freq

        # For now use scipy - needs scipy
        # hardcoded values should be exposed in some meaningful way

        import scipy.signal as signal
        peak_ind, props = signal.find_peaks(-np.abs(resp), distance=10, 
            prominence=.05)

        if make_plot:
            self.plot_find_peak(freq, resp_input, peak_ind, save_plot=save_plot,
                save_name=save_name)

        return freq[peak_ind]

    def plot_find_peak(self, freq, resp, peak_ind, save_plot=True, 
        save_name=None):
        """
        """
        import matplotlib.pyplot as plt

        Idat = np.real(resp)
        Qdat = np.imag(resp)
        phase = np.unwrap(np.arctan2(Qdat, Idat))
        
        fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))
        ax[0].plot(freq, np.abs(resp), label='amp', color='b')
        ax[0].plot(freq, Idat, label='I', color='r', linestyle=':', alpha=.5)
        ax[0].plot(freq, Qdat, label='Q', color='g', linestyle=':', alpha=.5)
        ax[0].legend(loc='lower right')
        ax[1].plot(freq, phase, color='b')
        ax[1].set_ylim((-np.pi, np.pi))

        if len(peak_ind):  # empty array returns False
            ax[0].plot(freq[peak_ind], np.abs(resp[peak_ind]), 'x', color='k')
            ax[1].plot(freq[peak_ind], phase[peak_ind], 'x', color='k')
        else:
            self.log('No peak_ind values.', self.LOG_USER)

        fig.suptitle("Peak Finding")
        ax[1].set_xlabel("Frequency offset from Subband Center (MHz)")
        ax[0].set_ylabel("Response")
        ax[1].set_ylabel("Phase [rad]")

        if save_plot:
            if save_name is None:
                self.log('Using default name for saving: find_peak.png \n' +
                    'Highly recommended that you input a non-default name')
                save_name = 'find_peak.png'
            else:
                self.log('Plotting saved to {}'.format(save_name))
            plt.savefig(os.path.join(self.plot_dir, save_name),
                bbox_inches='tight')
            plt.close()

    def find_all_peak(self, freq, resp, subband, normalize=False, 
        n_samp_drop=1, threshold=.5, margin_factor=1., phase_min_cut=1, 
        phase_max_cut=1):
        """
        find the peaks within each subband requested from a fullbandamplsweep

        Args:
        -----
        freqs (array):  (n_subbands x n_freqs_swept) array of frequencies swept
        response (complex array): n_subbands x n_freqs_swept array of complex 
            response
        subbands (list of ints): subbands that we care to search in

        Optional Args:
        --------------
        normalize (bool) : 
        n_samp_drop (int) :
        threshold (float) :
        margin_factor (float):
        phase_min_cut (int) :
        phase_max_cut (int) :
        """
        peaks = np.array([])
        subbands = np.array([])

        for sb in subband:
            peak = self.find_peak(freq[sb,:], resp[sb,:], 
                normalize=normalize, n_samp_drop=n_samp_drop, 
                threshold=threshold, margin_factor=margin_factor,
                phase_min_cut=phase_min_cut, phase_max_cut=phase_max_cut,
                make_plot=True, save_plot=True,
                save_name='find_peak_subband{:03}.png'.format(int(sb)))

            if peak is not None:
                peaks = np.append(peaks, peak)
                subbands = np.append(subbands, 
                    np.ones_like(peak, dtype=int)*sb)

        res = np.vstack((peaks, subbands))
        return res

    def fast_eta_scan(self, band, subband, freqs, n_read, drive, 
        make_plot=False):
        """copy of fastEtaScan.m from Matlab. Sweeps quickly across a range of
        freqs and gets I, Q response

        Args:
         band (int): which 500MHz band to scan
         subband (int): which subband to scan
         freqs (n_freqs x 1 array): frequencies to scan relative to subband 
            center
         n_read (int): number of times to scan
         drive (int): tone power

        Optional Args:
        make_plot (bool): Make eta plots

        Outputs:
         resp (n_freqs x 2 array): real, imag response as a function of 
            frequency
         freqs (n_freqs x n_read array): frequencies scanned, relative to 
            subband center
        """
        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        channel_order = self.get_channel_order(None) # fix this later

        channels_per_subband = int(n_channels / n_subbands)
        first_channel_per_subband = channel_order[0::channels_per_subband]
        subchan = first_channel_per_subband[subband]

        self.set_eta_scan_freqs(band, freqs)
        self.set_eta_scan_amplitude(band, drive)
        self.set_eta_scan_channel(band, subchan)
        self.set_eta_scan_dwell(band, 0)

        self.set_run_eta_scan(band, 1)

        I = self.get_eta_scan_results_real(band, count=len(freqs))
        Q = self.get_eta_scan_results_imag(band, count=len(freqs))

        band_off(epics_root, band)

        response = np.zeros((len(freqs), ), dtype=complex)

        for index in range(len(freqs)):
            Ielem = I[index]
            Qelem = Q[index]
            if Ielem > 2**23:
                Ielem = Ielem - 2**24
            if Qelem > 2**23:
                Qelem = Qelem - 2**24
            
            Ielem = Ielem / 2**23
            Qelem = Qelem / 2**23

            response[index] = Ielem + 1j*Qelem

        if make_plot:
            import matplotlib.pyplot as plt

        return response, freqs