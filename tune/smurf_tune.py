import numpy as np
from pysmurf.base import SmurfBase

class SmurfTuneMixin(SmurfBase):
    '''
    This contains all the tuning scripts
    '''

    def fast_eta_scan(self, band, subband, freqs, n_read, drive):
        """copy of fastEtaScan.m from Matlab. Sweeps quickly across a range of
        freqs and gets I, Q response

        Args:
         band (int): which 500MHz band to scan
         subband (int): which subband to scan
         freqs (n_freqs x 1 array): frequencies to scan relative to subband center
         n_read (int): number of times to scan
         drive (int): tone power

        Outputs:
         resp (n_freqs x 2 array): real, imag response as a function of frequency
         freqs (n_freqs x n_read array): frequencies scanned, relative to subband
            center
        """
        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        channelorder = self.get_channel_order(None) # fix this later

        channels_per_subband = int(n_channels / n_subbands)
        first_channel_per_subband = channelorder[0::channels_per_subband]
        subchan = first_channel_per_subband[subband]

        cryo_path = baseroot + "CryoChannels:"

        self.set_eta_scan_freqs(band, freqs)
        self.set_eta_scan_amplitude(band, drive)
        self.set_eta_scan_channel(band, subchan)
        self.set_eta_scan_dwell(band, 0)

        self.set_run_eta_scan(band, 1)

        I = self.get_eta_scan_results_real(band, count = len(freqs))
        Q = self.get_eta_scan_results_imag(band, count = len(freqs))

        band_off(epics_root, band)

        
        #if I > 2**23:
        #    I = I - 2**24
        #if Q > 2**23:
        #    Q = Q - 2**24


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