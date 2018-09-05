import numpy as np
from pysmurf.base import SmurfBase

class SmurfTuneMixin(SmurfBase):
    '''
    This contains all the tuning scripts
    '''


    def find_frequencies(self, band, subband=np.array([63]), drive_power=10,
        n_read=2):
        '''

        Optional Args:
        --------------
        drive_power (int) : The drive amplitude
        n_read (int) : The number sweeps to do per subband
        '''

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        n_subchannels = n_channels / n_subbands # 16

        band_center = self.get_band_center_mhz(band)


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


        return response, freqs