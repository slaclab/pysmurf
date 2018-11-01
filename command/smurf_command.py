import numpy as np
import os
import epics
import time
from pysmurf.base import SmurfBase

class SmurfCommandMixin(SmurfBase):

    def _caput(self, cmd, val, write_log=False, execute=True, wait_before=None,
        wait_after=None, wait_done=True, log_level=0):
        '''
        Wrapper around pyrogue lcaput. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        val: The value to put into epics

        Optional Args:
        --------------
        write_log (bool) : Whether to log the data or not. Default False
        execute (bool) : Whether to actually execute the command. Defualt True.
        wait_before (int) : If not None, the number of seconds to wait before
            issuing the command
        wait_after (int) : If not None, the number of seconds to wait after
            issuing the command
        wait_done (bool) : Wait for the command to be finished before returning.
            Default True.
        '''
        if wait_before is not None:
            if write_log:
                self.log('Waiting {:3.2f} seconds before...'.format(wait_before),
                    self.LOG_USER)
            time.sleep(wait_before)

        if write_log:
            log_str = 'caput ' + cmd + ' ' + str(val)
            if self.offline:
                log_str = 'OFFLINE - ' + log_str
            self.log(log_str, log_level)

        if execute and not self.offline:
            epics.caput(cmd, val, wait=wait_done)

        if wait_after is not None:
            if write_log:
                self.log('Waiting {:3.2f} seconds after...'.format(wait_after),
                    self.LOG_USER)
            time.sleep(wait_after)
            self.log('Done waiting.', log_level)

    def _caget(self, cmd, write_log=False, execute=True, count=None,
        log_level=0):
        '''
        Wrapper around pyrogue lcaget. Gets variables from epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        write_log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        count (int or None) : number of elements to return for array data

        Returns:
        --------
        ret : The requested value
        '''
        if write_log:
            self.log('caput ' + cmd, log_level)

        if execute and not self.offline:
            ret = epics.caget(cmd, count=count)
            if write_log:
                self.log(ret)
            return ret
        else:
            return None

    _number_sub_bands = 'numberSubBands'
    def get_number_sub_bands(self, band, **kwargs):
        '''
        Returns the number of subbands in a band.
        To do - possibly hide this function.

        Args:
        -----
        band (int): The band to count

        Returns:
        --------
        n_subbands (int): The number of subbands in the band
        '''
        if self.offline:
            return 128
        else:
            return self._caget(self._band_root(band) + self._number_sub_bands, 
                **kwargs)

    _number_channels = 'numberChannels'
    def get_number_channels(self, band, **kwargs):
        '''
        Returns the number of channels in a band.

        Args:
        -----
        band (int): The band to count

        Returns:
        --------
        n_channels (int): The number of channels in the band
        '''
        if self.offline:
            return 512
        else:
            return self._caget(self._band_root(band) + self._number_channels,
                **kwargs)

    def set_defaults_pv(self, **kwargs):
        '''
        Sets the default epics variables
        '''
        self._caput(self.epics_root + ':AMCc:setDefaults', 1, wait_after=5,
            **kwargs)
        self.log('Defaults are set.', self.LOG_INFO)

    def set_read_all(self, **kwargs):
        """
        ReadAll sends a command to read all register to the pyrogue server
        Registers must upated in order to PVs to update. 
        This call is necesary to read register with pollIntervale=0.
        """
        self._caput(self.epics_root + ':AMCc:ReadAll', 1, wait_after=5,
            **kwargs)
        self.log('ReadAll sent', self.LOG_INFO)


    _eta_scan_freqs = 'etaScanFreqs'
    def set_eta_scan_freq(self, band, val, **kwargs):
        '''
        Sets the frequency to do the eta scan

        Args:
        -----
        band (int) : The band to count
        val (int) :  The frequency to scan
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_freqs, val, 
            **kwargs)

    def get_eta_scan_freq(self, band, **kwargs):
        '''
        Args:
        -----
        band (int): The band to count

        Returns:
        freq (int) : The frequency of the scan
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_freqs, 
            **kwargs)

    _eta_scan_amplitude = 'etaScanAmplitude'
    def set_eta_scan_amplitude(self, band, val, **kwargs):
        '''
        Sets the amplitude of the eta scan.

        Args:
        -----
        band (int) : The band to set
        val (in) : The eta scan amplitude. Typical value is 9 to 11.
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_amplitude, val, 
            **kwargs)

    def get_eta_scan_amplitude(self, band, **kwargs):
        '''
        Gets the amplitude of the eta scan.

        Args:
        -----
        band (int) : The band to set

        Returns:
        --------
        amp (int) : The eta scan amplitude
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_amplitude, 
            **kwargs)

    _eta_scan_channel = 'etaScanChannel'
    def set_eta_scan_channel(self, band, val, **kwargs):
        '''
        Sets the channel to eta scan.

        Args:
        -----
        band (int) : The band to set
        val (int) : The channel to set
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_channel, val, 
            **kwargs)

    def get_eta_scan_channel(self, band, **kwargs):
        '''
        Gets the channel to eta scan.

        Args:
        -----
        band (int) : The band to set

        Returns:
        --------
        chan (int) : The channel that is being eta scanned
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_channel, 
            **kwargs)

    _eta_scan_dwell = 'etaScanDwell'
    def set_eta_scan_dwell(self, band, val, **kwargs):
        """
        Swets how long to dwell while eta scanning.

        Args:
        -----
        band (int) : The band to eta scan.
        val (int) : The time to dwell
        """
        self._caput(self._cryo_root(band) + self._eta_scan_dwell, val, **kwargs)

    def get_eta_scan_dwell(self, band, **kwargs):
        """
        Gets how long to dwell

        Args:
        -----
        band (int) : The band being eta scanned

        Returns:
        --------
        dwell (int) : The time to dwell during an eta scan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_dwell, 
            **kwargs)

    _run_eta_scan = 'runEtaScan'
    def set_run_eta_scan(self, band, val, **kwargs):
        """
        Runs the eta scan. Set the channel using set_eta_scan_channel()

        Args:
        -----
        band (int) : The band to eta scan. 
        val (bool) : Start the eta scan. 
        """
        self._caput(self._cryo_root(band) + self._run_eta_scan, val, **kwargs)

    def get_run_eta_scan(self, band, **kwargs):
        """
        Gets the status of eta scan.

        Args:
        -----
        band (int) : The band that is being checked

        Returns:
        --------
        status (int) : Whether the band is eta scanning.
        """
        return self._caget(self._cryo_root(band) + self._run_eta_scan, **kwargs)    

    _eta_scan_results_real = 'etaScanResultsReal'
    def get_eta_scan_results_real(self, band, count, **kwargs):
        '''
        Gets the real component of the eta scan.

        Args:
        -----
        band (int) : The to get eta scans
        count (int) : The number of samples to read

        Returns:
        --------
        resp (float array) : THe real component of the most recent eta scan
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_results_real,
            count=count, **kwargs)

    _eta_scan_results_imag = 'etaScanResultsImag'
    def get_eta_scan_results_imag(self, band, count, **kwargs):
        '''
        Gets the imaginary component of the eta scan.

        Args:
        -----
        band (int) : The to get eta scans
        count (int) : The number of samples to read

        Returns:
        --------
        resp (float array) : THe imaginary component of the most recent eta scan
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_results_imag,
            count=count, **kwargs)

    _amplitude_scales = 'setAmplitudeScales'
    def set_amplitude_scales(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._amplitude_scales, val, 
            **kwargs)

    def get_amplitude_scales(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._amplitude_scales,
            **kwargs)

    _amplitude_scale_array = 'amplitudeScaleArray'
    def set_amplitude_scale_array(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._amplitude_scale_array, val,
            **kwargs)

    def get_amplitude_scale_array(self, band, **kwargs):
        '''
        Gets the array of amplitudes

        Args:
        -----
        band (int) : The band to search.

        Returns:
        --------
        amplitudes (array) : The tone amplitudes
        '''
        return self._caget(self._cryo_root(band) + self._amplitude_scale_array, 
            **kwargs)

    _feedback_enable_array = 'feedbackEnableArray'
    def set_feedback_enable_array(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._feedback_enable_array, val,
            **kwargs)

    def get_feedback_enable_array(self, band, **kwargs):
        '''
        Gets the array of feedbacks enables

        Args:
        -----
        band (int) : The band to search.

        Returns:
        --------
        fb_on (boolean array) : An array of whether the feedback is on or off.
        '''
        return self._caget(self._cryo_root(band) + self._feedback_enable_array, 
            **kwargs)

    _single_channel_readout = 'singleChannelReadout'
    def set_single_channel_readout(self, band, val, **kwargs):
        '''
        Sets the singleChannelReadout bit.

        Args:
        -----
        band (int): The band to set to single channel readout
        '''
        self._caput(self._band_root(band) + self._single_channel_readout, val, 
            **kwargs)

    def get_single_channel_readout(self, band, **kwargs):
        '''

        '''
        return self._caget(self._band_root(band) + self._single_channel_readout, 
            **kwargs)

    _single_channel_readout2 = 'singleChannelReadoutOpt2'
    def set_single_channel_readout_opt2(self, band, val, **kwargs):
        '''
        Sets the singleChannelReadout2 bit.

        Args:
        -----
        band (int): The band to set to single channel readout
        '''
        self._caput(self._band_root(band) + self._single_channel_readout2, val, 
            **kwargs)

    def get_single_channel_readout_opt2(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._single_channel_readout2, 
            **kwargs)

    _stream_enable = 'streamEnable'
    def set_stream_enable(self, band, val, **kwargs):
        """
        Enable/disable streaming data
        """
        self._caput(self._band_root(band) + self._stream_enable, val, **kwargs)

    def get_stream_enable(self, band, **kwargs):
        """
        Enable/disable streaming data
        """
        return self._caget(self._band_root(band) + self._stream_enable, 
            **kwargs)

    _iq_stream_enable = 'iqStreamEnable'
    def set_iq_stream_enable(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._iq_stream_enable, val, **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band)  + self._iq_stream_enable, 
            **kwargs)

    _decimation = 'decimation'
    def set_decimation(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._decimation, val, **kwargs)

    def get_decimation(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._decimation, **kwargs)

    _filter_alpha = 'filterAlpha'
    def set_filter_alpha(self, band, val, **kwargs):
        '''
        Coefficient for single pole low pass fitler before readout (when c
        hannels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize 
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        '''
        self._caput(self._band_root(band) + self._filter_alpha, val, **kwargs)

    def get_filter_alpha(self, band, **kwargs):
        '''
        Coefficient for single pole low pass fitler before readout (when 
        channels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize 
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        '''
        return self._caget(self._band_root(band) + self._filter_alpha, **kwargs)

    _iq_swap_in = 'iqSwapIn'
    def set_iq_swap_in(self, band, val, **kwargs):
        '''
        Swaps I&Q into DSP (from ADC).  Tones being output by the system will 
        flip about the band center (e.g. 4.25GHz, 5.25GHz etc.)
        '''
        self._caput(self._band_root(band) + self._iq_swap_in, val, **kwargs)

    def get_iq_swap_in(self, band, **kwargs):
        '''
        Swaps I&Q into DSP (from ADC).  Tones being output by the system will 
        flip about the band center (e.g. 4.25GHz, 5.25GHz etc.)
        '''
        return self._caget(self._band_root(band) + self._iq_swap_in, **kwargs)

    _iq_swap_out = 'iqSwapOut'
    def set_iq_swap_out(self, band, val, **kwargs):
        '''
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum around band 
        center.
        '''
        self._caput(self._band_root(band) + self._iq_swap_out, val, **kwargs)

    def get_iq_swap_out(self, band, **kwargs):
        '''
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum around band 
        center.
        '''
        return self._caget(self._band_root(band) + self._iq_swap_out, **kwargs)

    _ref_phase_delay = 'refPhaseDelay'
    def set_ref_phase_delay(self, band, val, **kwargs):
        '''
        Corrects for roundtrip cable delay
        freqError = IQ * etaMag, rotated by etaPhase+refPhaseDelay
        '''
        self._caput(self._band_root(band) + self._ref_phase_delay, val, 
            **kwargs)

    def get_ref_phase_delay(self, band, **kwargs):
        '''
        Corrects for roundtrip cable delay
        freqError = IQ * etaMag, rotated by etaPhase+refPhaseDelay
        '''
        return self._caget(self._band_root(band) + self._ref_phase_delay, 
            **kwargs)

    _ref_phase_delay_fine = 'refPhaseDelayFine'
    def set_ref_phase_delay_fine(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._ref_phase_delay_fine, val, 
        **kwargs)

    def get_ref_phase_delay_fine(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._ref_phase_delay_fine, 
            **kwargs)

    _tone_scale = 'toneScale'
    def set_tone_scale(self, band, val, **kwargs):
        '''
        Scales the sum of 16 tones before synthesizer.
        '''
        self._caput(self._band_root(band) + self._tone_scale, val, **kwargs)

    def get_tone_scale(self, band, **kwargs):
        '''
        Scales the sum of 16 tones before synthesizer.
        '''
        return self._caget(self._band_root(band) + self._tone_scale, **kwargs)

    _waveform_select = 'waveformSelect'
    def set_waveform_select(self, band, val, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        self._caput(self._band_root(band) + self._waveform_select, val, 
            **kwargs)

    def get_waveform_select(self, band, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        return self._caget(self._band_root(band) + self._waveform_select, 
            **kwargs)

    _waveform_start = 'waveformStart'
    def set_waveform_start(self, band, val, **kwargs):
        """
        0x1 enables waveform table
        """
        self._caput(self._band_root(band) + self._waveform_start, val, 
            **kwargs)

    def get_waveform_start(self, band, **kwargs):
        """
        0x1 enables waveform table
        """
        return self._caget(self._band_root(band) + self._waveform_start, 
            **kwargs)    

    _rf_enable = 'rfEnable'
    def set_rf_enable(self, band, val, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        self._caput(self._band_root(band) + self._rf_enable, val, 
            **kwargs)

    def get_rf_enable(self, band, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        return self._caget(self._band_root(band) + self._rf_enable, **kwargs) 

    _analysis_scale = 'analysisScale'
    def set_analysis_scale(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._analysis_scale, val, **kwargs)

    def get_analysis_scale(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._analysis_scale, 
            **kwargs)

    _feedback_enable = 'feedbackEnable'
    def set_feedback_enable(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._feedback_enable, val, 
            **kwargs)

    def get_feedback_enable(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._feedback_enable, 
            **kwargs)

    _center_frequency_array = 'centerFrequencyArray'
    def set_center_frequency_array(self, band, val, **kwargs):
        """
        Sets all the center frequencies in a band
        """
        self._caput(self._cryo_root(band) + self._center_frequency_array, val,
            **kwargs)

    def get_center_frequency_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._center_frequency_array, 
            **kwargs)

    _feedback_gain = 'feedbackGain'
    def set_feedback_gain(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._feedback_gain, val, **kwargs)

    def get_feedback_gain(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._feedback_gain, 
            **kwargs)

    _eta_phase_array = 'etaPhaseArray'
    def set_eta_phase_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._eta_phase_array, val, 
            **kwargs)

    def get_eta_phase_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_phase_array, 
            **kwargs)

    _eta_mag_array = 'etaMagArray'
    def set_eta_mag_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._eta_mag_array, val, **kwargs)

    def get_eta_mag_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_mag_array, 
            **kwargs)

    _feedback_limit = 'feedbackLimit'
    def set_feedback_limit(self, band, val, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        self._caput(self._band_root(band) + self._feedback_limit, val, **kwargs)

    def get_feedback_limit(self, band, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        return self._caget(self._band_root(band) + self._feedback_limit, 
            **kwargs)

    _noise_select = 'noiseSelect'
    def set_noise_select(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._noise_select, val, 
            **kwargs)

    def get_noise_select(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._noise_select, 
            **kwargs)


    _lms_delay = 'lmsDelay'
    def set_lms_delay(self, band, val, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_delay, val, **kwargs)

    def get_lms_delay(self, band, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_delay, **kwargs)

    _lms_gain = 'lmsGain'
    def set_lms_gain(self, band, val, **kwargs):
        '''
        LMS gain, powers of 2
        '''
        self._caput(self._band_root(band) + self._lms_gain, val, **kwargs)

    def get_lms_gain(self, band, **kwargs):
        '''
        LMS gain, powers of 2
        '''
        return self._caget(self._band_root(band) + self._lms_gain, **kwargs)

    _lms_enable1 = 'lmsEnable1'
    def set_lms_enable1(self, band, val, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable1, val, **kwargs)

    def get_lms_enable1(self, band, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable1, **kwargs)

    _lms_enable2 = 'lmsEnable2'
    def set_lms_enable2(self, band, val, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable2, val, **kwargs)

    def get_lms_enable2(self, band, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable2, **kwargs)

    _lms_enable3 = 'lmsEnable3'
    def set_lms_enable3(self, band, val, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable3, val, **kwargs)

    def get_lms_enable3(self, band, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable3, **kwargs)

    _lms_rst_dly = 'lmsRstDly'
    def set_lms_rst_dly(self, band, val, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_rst_dly, val, **kwargs)

    def get_lms_rst_dly(self, band, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_rst_dly, **kwargs)

    _lms_freq = 'lmsFreq'
    def set_lms_freq(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(self._band_root(band) + self._lms_freq, val, **kwargs)

    def get_lms_freq(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(self._band_root(band) + self._lms_freq, **kwargs)

    _lms_freq_hz = 'lmsFreqHz'
    def set_lms_freq_hz(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(self._band_root(band) + self._lms_freq_hz, val, **kwargs)

    def get_lms_freq_hz(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(self._band_root(band) + self._lms_freq_hz, **kwargs)

    _lms_dly_fine = 'lmsDlyFine'
    def set_lms_dly_fine(self, band, val, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_dly_fine, val, **kwargs)

    def get_lms_dly_fine(self, band, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_dly_fine, **kwargs)

    _lms_delay2 = 'lmsDelay2'
    def set_lms_delay2(self, band, val, **kwargs):
        """
        delay DDS counter reset (307.2MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_delay2, val, **kwargs)

    def get_lms_delay2(self, band, **kwargs):
        """
        delay DDS counter reset (307.2MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_delay2, **kwargs)

    _iq_stream_enable = 'iqStreamEnable'
    def set_iq_stream_enable(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._iq_stream_enable, val, 
            **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._iq_stream_enable, 
            **kwargs)

    _feedback_polarity = 'feedbackPolarity'
    def set_feedback_polarity(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._feedback_polarity, val, 
            **kwargs)

    def get_feedback_polarity(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._feedback_polarity, 
            **kwargs)

    _band_center_mhz = 'bandCenterMHz'
    def set_band_center_mhz(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._band_center_mhz, val, 
            **kwargs)

    def get_band_center_mhz(self, band, **kwargs):
        '''
        '''
        if self.offline:
            if band == 3:
                bc = 5.25E3
            elif band == 2:
                bc = 5.75E3
            return bc
        else:
            return self._caget(self._band_root(band) + self._band_center_mhz, 
                **kwargs)

    _digitizer_frequency_mhz = 'digitizerFrequencyMHz'
    def set_digitizer_frequency_mhz(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._digitizer_frequency_mhz, val, 
            **kwargs)

    def get_digitizer_frequency_mhz(self, band, **kwargs):
        '''
        '''
        if self.offline:
            return 614.4
        else:
            return self._caget(self._band_root(band) + 
                self._digitizer_frequency_mhz, **kwargs)

    _synthesis_scale = 'synthesisScale'
    def set_synthesis_scale(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._synthesis_scale, val, 
            **kwargs)

    def get_synthesis_scale(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._synthesis_scale, 
            **kwargs)

    _dsp_enable = 'dspEnable'
    def set_dsp_enable(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._dsp_enable, val, **kwargs)

    def get_dsp_enable(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._dsp_enable, **kwargs)

    # Single channel commands
    _amplitude_scale_channel = 'amplitudeScale'
    def set_amplitude_scale_channel(self, band, channel, val, **kwargs):
        """
        Set the amplitude scale for a single channel
        """
        self._caput(self._channel_root(band, channel) + 
            self._amplitude_scale_channel, val, **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        """
        """
        self._caget(self._channel_root(band, channel) + 
            self._amplitude_scale_channel, **kwargs)

    _feedback_enable = 'feedbackEnable'
    def set_feedback_enable_channel(self, band, channel, val, **kwargs):
        """
        Set the feedback for a single channel
        """
        self._caput(self._channel_root(band, channel) + 
            self._feedback_enable, val, **kwargs)

    def get_feedback_enable_channel(self, band, channel, **kwargs):
        """
        Get the feedback for a single channel
        """
        self._caget(self._channel_root(band, channel) + 
            self._feedback_enable, **kwargs)

    _eta_mag_scaled_channel = 'etaMagScaled'
    def set_eta_mag_scaled_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) + 
            self._eta_mag_scaled_channel, val, **kwargs)

    def get_eta_mag_scaled_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._eta_mag_scaled_channel, **kwargs)


    _center_frequency_mhz_channel = 'centerFrequencyMHz'
    def set_center_frequency_mhz_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) + 
            self._center_frequency_mhz_channel, val, **kwargs)

    def get_center_frequency_mhz_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._center_frequency_mhz_channel, **kwargs)


    _amplitude_scale_channel = 'amplitudeScale'
    def set_amplitude_scale_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) + 
            self._amplitude_scale_channel, val, **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) + 
            self._amplitude_scale_channel, **kwargs)

    _eta_phase_degree_channel = 'etaPhaseDegree'
    def set_eta_phase_degree_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) +
            self._eta_phase_degree_channel, val, **kwargs)

    def get_eta_phase_degree_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._eta_phase_degree_channel, **kwargs)

    _frequency_error_mhz = 'frequencyErrorMHz'
    def get_frequency_error_mhz(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._frequency_error_mhz, **kwargs)


    # Attenuator
    _uc = 'UC[{}]'
    def set_att_uc(self, b, val, input_band=True, **kwargs):
        '''
        Set the upconverter attenuator

        Args:
        -----
        b (int): Either the band or the attenuator number.
        val (int): The attenuator value

        Opt Args:
        ---------
        input_band (bool): If True, the input is assumed to be the band number.
            Otherwise the input is assumed to be the attenuator number
        '''
        if input_band:
            b = self.band_to_att(b)
        self._caput(self.att_root + self._uc.format(int(b)), val, **kwargs)

    def get_att_uc(self, b, input_band=True, **kwargs):
        '''
        Get the upconverter attenuator value
        '''
        if input_band:
            b = self.band_to_att(b)
        return self._caget(self.att_root + self._uc.format(int(b)), **kwargs)


    _dc = 'DC[{}]'
    def set_att_dc(self, b, val, input_band=True, **kwargs):
        '''
        Set the down-converter attenuator

        Args:
        -----
        b (int): Either the band or the attenuator number.
        val (int): The attenuator value

        Opt Args:
        ---------
        input_band (bool): If True, the input is assumed to be the band number.
            Otherwise the input is assumed to be the attenuator number
        '''
        if input_band:
            b = self.band_to_att(b)
        self._caput(self.att_root + self._dc.format(int(b)), val, **kwargs)

    def get_att_dc(self, b, input_band=True, **kwargs):
        '''
        Get the down-converter attenuator value
        '''
        if input_band:
            b = self.band_to_att(b)
        return self._caget(self.att_root + self._dc.format(int(b)), **kwargs)

    # ADC commands
    _adc_remap = "Remap[0]"  # Why is this hardcoded 0
    def set_remap(self, **kwargs):
        '''
        This command should probably be renamed to something more descriptive.
        '''
        self._caput(self.adc_root + self._adc_remap, 1, **kwargs)

    # DAC commands
    _dac_enable = "enable"
    def set_dac_enable(self, b, val, **kwargs):
        '''
        Enables DAC
        '''
        self._caput(self.dac_root.format(b) + self._dac_enable, val, **kwargs)

    def get_dac_enable(self, b, **kwargs):
        '''
        Gets enable status of DAC
        '''
        self._caget(self.dac_root.format(b) + self._dac_enable, **kwargs)

    # Jesd commands
    _data_out_mux = 'dataOutMux[{}]'
    def set_data_out_mux(self, b, val, **kwargs):
        '''
        '''
        self._caput(self.jesd_tx_root + 
            self._data_out_mux.format(b), val, **kwargs)

    def get_data_out_mux(self, b, **kwargs):
        '''
        '''
        return self._caget(self.jesd_tx_root + self._data_out_mux.format(b), 
            val, **kwargs)

    _link_disable = 'LINK_DISABLE'
    def set_jesd_link_disable(self, val, **kwargs):
        '''
        Disables jesd link
        '''
        self._caput(self.jesd_rx_root + self._link_disable, val, **kwargs)

    def get_jesd_link_disable(self, **kwargs):
        '''
        Disables jesd link
        '''
        return self._caget(self.jesd_rx_root + self._link_disable, val, 
            **kwargs)

    _jesd_tx_enable = 'Enable'
    def get_jesd_tx_enable(self, **kwargs):
        '''
        '''
        return self._caget(self.jesd_tx_root + self._jesd_tx_enable, **kwargs)

    _jesd_tx_valid = 'DataValid'
    def get_jesd_tx_data_valid(self, **kwargs):
        return self._caget(self.jesd_tx_root + self._jesd_tx_enable, **kwargs)

    # _start_addr = 'StartAddr[{}]'
    # def set_start_addr(self, b, val, **kwargs):
    #     """
    #     """
    #     self._caput(self.waveform_engine_buffers_root + \
    #         self._start_addr.format(b), val, **kwargs)

    # def get_start_addr(self, val, **kwargs):
    #     """
    #     """
    #     return self._caget(self.waveform_engine_buffers_root + \
    #         self._start_addr.format(b), **kwargs)

    _fpga_uptime = 'UpTimeCnt'
    def get_fpga_uptime(self, **kwargs):
        '''
        Returns:
        uptime (float) : The FPGA uptime
        '''
        return self._caget(self.axi_version + self._fpga_uptime, **kwargs)

    _fpga_version = 'FpgaVersion'
    def get_fpga_version(self, **kwargs):
        '''
        Returns:
        version (str) : The FPGA version
        '''
        return self._caget(self.axi_version + self._fpga_version, **kwargs)

    _fpga_git_hash = 'GitHash'
    def get_fpga_git_hash(self, **kwargs):
        '''
        Returns:
        git_hash (str) : The git has of the FPGA
        '''
        return self._caget(self.axi_version + self._fpga_git_hash, **kwargs)

    _fpga_build_stamp = 'BuildStamp'
    def get_fpga_build_stamp(self, **kwargs):
        '''
        Returns:
        build_stamp (str) : The FPGA build stamp
        '''
        return self._caget(self.axi_version + self._fpga_build_stamp, **kwargs)

    _input_mux_sel = 'InputMuxSel[{}]'
    def set_input_mux_sel(self, b, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root + self._input_mux_sel.format(b), val,
            **kwargs)

    def get_input_mux_sel(self, b, **kwargs):
        """
        """
        self._caget(self.daq_mux_root + self._input_mux_sel.format(b), 
            **kwargs)

    _data_buffer_size = 'DataBufferSize'
    def set_data_buffer_size(self, val, **kwargs):
        """
        Sets the data buffer size for the DAQx
        """
        self._caput(self.daq_mux_root + self._data_buffer_size, val, **kwargs)

    def get_data_buffer_size(self, **kwargs):
        """
        Gets the data buffer size for the DAQs
        """
        return self._caget(self.daq_mux_root + self._data_buffer_size, 
            **kwargs)

    # Waveform engine commands
    _start_addr = 'StartAddr[{}]'
    def set_waveform_start_addr(self, b, val, convert=True, **kwargs):
        """
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root + 
            self._start_addr.format(b), val, **kwargs)

    def get_waveform_start_addr(self, b, convert=True, **kwargs):
        """

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        val = self._caget(self.waveform_engine_buffers_root + 
            self._start_addr.format(b), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _end_addr = 'EndAddr[{}]'
    def set_waveform_end_addr(self, b, val, convert=True, **kwargs):
        """

        Optional Arg:
        -------------
        convert (bool) : Convert the input from an int to a string of hex
            values. Default (True)
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root + 
            self._end_addr.format(b), val, **kwargs)

    def get_waveform_end_addr(self, b, convert=True, **kwargs):
        """
        """
        val = self._caget(self.waveform_engine_buffers_root + 
            self._end_addr.format(b), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val
    
    _wr_addr = 'WrAddr[{}]'
    def set_waveform_wr_addr(self, b, val, convert=True, **kwargs):
        """
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root + 
            self._wr_addr.formmat(b), val, **kwargs)

    def get_waveform_wr_addr(self, b, convert=True, **kwargs):
        """
        """
        val = self._caget(self.waveform_engine_buffers_root + 
            self._wr_addr.format(b), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _empty = 'Empty[{}]'
    def set_waveform_empty(self, b, val, **kwargs):
        """
        """
        self._caput(self.waveform_engine_buffers_root + 
            self._empty.format(b), **kwargs)

    def get_waveform_empty(self, b, **kwargs):
        """
        """
        return self._caget(self.waveform_engine_buffers_root + 
            self._empty.format(b), **kwargs)

    _data_file = 'dataFile'
    def set_streamdatawriter_datafile(self, val, **kwargs):
        """
        """
        self._caput(self.stream_data_writer_root + self._data_file,
            val, **kwargs)

    def get_streamdatawriter_datafile(self, **kwargs):
        """
        """
        return self._caget(self.stream_data_writer_root + 
            self._data_file, **kwargs)

    _datawriter_open = 'open'
    def set_streamdatawriter_open(self, val, **kwargs):
        """
        """
        self._caput(self.stream_data_writer_root + 
            self._datawriter_open, val, **kwargs)

    def get_streamdatawriter_open(self, **kwargs):
        """
        """
        return self._caget(self.stream_data_writer_root + 
            self._datawriter_open, **kwargs)


    _trigger_daq = 'TriggerDaq'
    def set_trigger_daq(self, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root + self._trigger_daq, val, 
            **kwargs)

    def get_trigger_daq(self, **kwargs):
        """
        """
        self._caget(self.daq_mux_root + self._trigger_daq, 
            **kwargs)

    _arm_hw_trigger = "ArmHwTrigger"
    def set_arm_hw_trigger(self, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root + self._arm_hw_trigger, val, **kwargs)

    _trigger_hw_arm = 'TriggerHwArm'
    def set_trigger_hw_arm(self, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root + self._trigger_hw_arm, val, **kwargs)

    def get_trigger_hw_arm(self, **kwargs):
        """
        """
        return self._caget(self.daq_mux_root + self._trigger_hw_arm, **kwargs)

    # rtm commands
    _reset_rtm = 'resetRtm'
    def reset_rtm(self, **kwargs):
        '''
        Resets the rear transition module (RTM)
        '''
        self._caput(self.rtm_cryo_det_root + self._reset_rtm, 1, **kwargs)

    _cpld_reset = 'CpldReset'
    def set_cpld_reset(self, val, **kwargs):
        """
        Args:
        -----
        val (int) : Set to 1 for a cpld reset
        """
        self._caput(self.rtm_cryo_det_root + self._cpld_reset, val, **kwargs)    

    def get_cpld_reset(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._cpld_reset, **kwargs)

    def cpld_toggle(self, **kwargs):
        """
        Toggles the cpld reset bit.
        """
        self.set_cpld_reset(1, wait_done=True, **kwargs)
        self.set_cpld_reset(0, wait_done=True, **kwargs)

    _low_cycle = 'LowCycle'
    def set_low_cycle(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._low_cycle, val, **kwargs)

    def get_low_cycle(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._low_cycle, **kwargs)

    _high_cycle = 'HighCycle'
    def set_high_cycle(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._high_cycle, val, **kwargs)

    def get_high_cycle(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._high_cycle, **kwargs)

    _k_relay = 'KRelay'
    def set_k_relay(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._k_relay, val, **kwargs)

    def get_k_relay(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._k_relay, **kwargs)

    _ramp_max_cnt = 'RampMaxCnt'
    def set_ramp_max_cnt(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_max_cnt, val, **kwargs)

    def get_ramp_max_cnt(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_max_cnt, 
            **kwargs)

    _select_ramp = 'SelectRamp'
    def set_select_ramp(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._select_ramp, val, **kwargs)

    def get_select_ramp(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_dte_root + self._select_ramp, **kwargs)

    _ramp_start_mode = 'RampStartMode'
    def set_ramp_start_mode(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_start_mode, val,
            **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_start_mode, 
            **kwargs)

    _pulse_width = 'PulseWidth'
    def set_pulse_width(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._pulse_width, val, **kwargs)

    def get_pulse_width(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._pulse_width, **kwargs)

    _debounce_width = 'DebounceWidth'
    def set_debounce_width(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._debounce_width, val, 
            **kwargs)

    def get_debounce_width(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._debounce_width, 
            **kwargs)

    _ramp_slope = 'RampSlope'
    def set_ramp_slope(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._ramp_slope, val, **kwargs)

    def get_ramp_slope(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._ramp_slope, **kwargs)

    _mode_control = 'ModeControl'
    def set_mode_control(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._mode_control, val, **kwargs)

    def get_mode_control(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._mode_control, **kwargs)

    _fast_slow_step_size = 'FastSlowStepSize'
    def set_fast_slow_step_size(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._fast_slow_step_size, val, 
            **kwargs)

    def get_fast_slow_step_size(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._fast_slow_step_size, 
            **kwargs)

    _fast_slow_rst_value = 'FastSlowRstValue'
    def set_fast_slow_rst_value(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._fast_slow_rst_value, val, 
            **kwargs)

    def get_fast_slow_rst_value(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._fast_slow_rst_value, 
            **kwargs)
    
    _enable_ramp_trigger = 'EnableRampTrigger'
    def set_enable_ramp_trigger(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._enable_ramp_trigger, val,
            **kwargs)

    def get_enable_ramp_trigger(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._enable_ramp_trigger,
            **kwargs)

    _cfg_reg_ena_bit = 'CfgRegEnaBit'
    def set_cfg_reg_ena_bit(self, val, **kwargs):
        '''
        '''
        self._caput(self.rtm_spi_root + self._cfg_reg_ena_bit, val, **kwargs)

    def get_cfg_reg_ena_bit(self, **kwargs):
        '''
        '''
        return self._caget(self.rtm_spi_root + self._cfg_reg_ena_bit, **kwargs)

    _tes_bias_enable = 'TesBiasDacCtrlRegCh[{}]'
    def set_tes_bias_enable(self, daq, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_max_root + self._tes_bias_enable.format(daq), 
            val, **kwargs)


    def get_tes_bias_enable(self, daq, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_max_root + self._tes_bias.format(daq), 
            **kwargs)


    _bit_to_V = 2.035/float(2**19)
    _dac_num_50k = 2
    def set_50k_amp_gate_voltage(self, voltage, **kwargs):
        """
        """
        if voltage > 0 or voltage < -1.:
            self.log('Voltage must be between -1 and 0. Doing nothing.')
        else:
            self.set_tes_bias(self._dac_num_50k, voltage/self._bit_to_V, 
                **kwargs)

    def get_50k_amp_gate_voltage(self, **kwargs):
        """
        """
        return self._bit_to_V * self.get_tes_bias(self._dac_num_50k, **kwargs)

    def set_50k_amp_enable(self, disable=False, **kwargs):
        """
        Sets the 50K amp bit to 2 for enable and 0 for disable. Also 
        automatically sets the gate voltage to 0.

        Opt Args:
        ---------
        disable (bool) : Disable the 50K amplifier. Default False.
        """
        self.set_50k_amp_gate_voltage(0, wait_done=True, **kwargs)
        if disable:
            self.set_tes_bias_enable(self._dac_num_50k, 0, **kwargs)
        else:
            self.set_tes_bias_enable(self._dac_num_50k, 2, **kwargs)

    _tes_bias = 'TesBiasDacDataRegCh[{}]'
    def set_tes_bias(self, daq, val, **kwargs):
        """
        Sets the TES bias current

        Args:
        -----
        val (int) : the TES bias current in unit
        """

        if val > 2**19-1:
            val = 2**19-1
            self.log('Bias too high. Must be <= than 2^19-1. Setting to ' + 
                'max value', self.LOG_ERROR)
        elif val < -2**19:
            val = -2**19
            self.log('Bias too low. Must be >= than -2^19. Setting to ' + 
                'min value', self.LOG_ERROR)
        self._caput(self.rtm_spi_max_root + self._tes_bias.format(daq), val, 
            **kwargs)


    def get_tes_bias(self, daq, **kwargs):
        """
        Gets the TES bias current

        Returns:
        --------
        bias (int) : The TES bias current
        """
        return self._caget(self.rtm_spi_max_root + self._tes_bias.format(daq), 
            **kwargs)

    _bit_to_volt = 10./2**19
    def set_tes_bias_volt(self, dac_num, val, **kwargs):
        """
        """
        self.set_tes_bias(dac_num, val/self._bit_to_volt, **kwargs)


    def get_tes_bias_volt(self, dac_num, **kwargs):
        """
        """
        return self._bit_to_volt * self.get_tes_bias(dac_num, **kwargs)

    def flux_ramp_on(self, **kwargs):
        '''
        Turns on the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        '''
        self.set_cfg_reg_ena_bit(1, **kwargs)

    def flux_ramp_off(self, **kwargs):
        '''
        Turns off the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        '''
        self.set_cfg_reg_ena_bit(0, **kwargs)

    _ramp_max_cnt = 'RampMaxCnt'
    def set_ramp_max_cnt(self, val, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_max_cnt, val, **kwargs)

    def get_ramp_max_cnt(self, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_max_cnt, 
            **kwargs)

    def set_flux_ramp_freq(self, val, **kwargs):
        """
        Wrapper function for set_ramp_max_cnt. Takes input in Hz.
        """
        cnt = 3.072E5/float(val)-1
        self.set_ramp_max_cnt(cnt, **kwargs)

    def get_flux_ramp_freq(self, **kwargs):
        """
        Returns flux ramp freq in units of Hz
        """
        return 3.0725E5/(self.get_ramp_max_cnt(**kwargs)+1)

    _low_cycle = 'LowCycle'
    def set_low_cylce(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive). 
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._low_cycle, val, **kwargs)
    
    def get_low_cylce(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive). 
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(self.rtm_cryo_det_root + self._low_cycle, **kwargs)
 
    _high_cycle = 'HighCycle'
    def set_high_cylce(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).
        Along with LowCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._high_cycle, val, **kwargs)
    
    def get_high_cylce(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).
        Along with LowCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(self.rtm_cryo_det_root + self._high_cycle, **kwargs)

    _select_ramp = 'SelectRamp'
    def set_select_ramp(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._select_ramp, val, **kwargs)

    def get_select_ramp(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._select_ramp, **kwargs)

    _enable_ramp = 'EnabletRamp'
    def set_enable_ramp(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._enable_ramp, val, **kwargs)

    def get_enable_ramp(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._enable_ramp, **kwargs)

    _ramp_start_mode = 'RampStartMode'
    def set_ramp_start_mode(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_start_mode, val, 
            **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_start_mode, 
            **kwargs)

    _pulse_width = 'PulseWidth'
    def set_pulse_width(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._pulse_width, val, **kwargs)

    def get_pulse_width(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._pulse_width, **kwargs)

    _hemt_v_enable = 'HemtBiasDacCtrlRegCh[33]'
    def set_hemt_enable(self, disable=False, **kwargs):
        """
        Sets bit to 2 for enable and 0 for disable. Also automatically sets 
        HEMT gate voltage to 0.

        Opt Args:
        ---------
        disable (bool): If True, sets the bit to 0. 
        """
        self.set_hemt_gate_voltage(0, wait_done=True, **kwargs)
        if disable:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 0, 
                **kwargs)
        else:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 2, 
                **kwargs)


    _bit_to_V_hemt = .576/3.0E5  # empirically found
    def set_hemt_gate_voltage(self, voltage, override=False, **kwargs):
        """
        Sets the HEMT gate voltage in units of volts. 

        Args:
        -----
        voltage (float): The voltage applied to the HEMT gate. Must be between
            0 and .75. 

        Opt Args:
        ---------
        override (bool): Override thee limits on HEMT gate voltage. Default
            False.
        """
        if (voltage > .75 or voltage < 0 ) and not override:
            self.log('Input voltage too high. Not doing anything.' + 
                ' If you really want it higher, use the override optinal arg.')
        else:
            self.set_hemt_bias(int(voltage/self._bit_to_V_hemt), 
                override=override, **kwargs)

    _hemt_v = 'HemtBiasDacDataRegCh[33]'
    def set_hemt_bias(self, val, override=False, **kwargs):
        '''
        Sets the HEMT voltage in units of bits. Need to figure out the
        conversion into real units.

        There is a hardcoded maximum value. If exceeded, no voltage is set. This
        check can be ignored using the override optional argument.

        Args:
        -----
        val (int) : The voltage in bits

        Optional Args:
        --------------
        override (bool) : Allows exceeding the hardcoded limit. Default False.
        '''
        if val > 350E3 and not override:
            self.log('Input voltage too high. Not doing anything.' + 
                ' If you really want it higher, use the override optinal arg.')
        else:
            self._caput(self.rtm_spi_max_root + self._hemt_v, val, **kwargs)

    def get_hemt_bias(self, **kwargs):
        '''
        Returns the HEMT voltage in bits.
        '''
        return self._caget(self.rtm_spi_max_root + self._hemt_v, **kwargs)

    def get_hemt_gate_voltage(self, **kwargs):
        '''
        Returns the HEMT voltage in bits.
        '''
        return self._bit_to_V_hemt*self.get_hemt_bias(**kwargs)


    _stream_datafile = 'dataFile'
    def set_streaming_datafile(self, datafile, as_string=True, **kwargs):
        """
        Sets the datafile to write streaming data

        Args:
        -----
        datafile (str or length 300 int array): The name of the datafile

        Opt Args:
        ---------
        as_string (bool): The input data is a string. If False, the input
            data must be a length 300 character int. Default True.
        """
        if as_string:
            datafile = [ord(x) for x in datafile]
            # must be exactly 300 elements long. Pad with trailing zeros
            datafile = np.append(datafile, np.zeros(300-len(datafile), 
                dtype=int))
        self._caput(self.streaming_root + self._stream_datafile, datafile, 
            **kwargs)

    def get_streaming_datafile(self, as_string=True, **kwargs):
        """
        Gets the datafile that streaming data is written to.

        Returns:
        --------
        datafile (str or length 300 int array): The name of the datafile

        Opt Args:
        ---------
        as_string (bool): The output data returns as a string. If False, the 
            input data must be a length 300 character int. Default True.
        """
        datafile = self._caget(self.streaming_root + self._stream_datafile, 
            **kwargs)
        if as_string:
            datafile = ''.join([chr(x) for x in datafile])
        return datafile

    _streaming_file_open = 'open'
    def set_streaming_file_open(self, val, **kwargs):
        """
        Sets the streaming file open. 1 for streaming on. 0 for streaming off.

        Args:
        -----
        val (int): The streaming status
        """
        self._caput(self.streaming_root + self._streaming_file_open, val, 
            **kwargs)

    def get_streaming_file_open(self, **kwargs):
        """
        Gets the streaming file status. 1 is streaming, 0 is not.

        Returns:
        --------
        val (int): The streaming status.
        """
        return self._caget(self.streaming_root + self._streaming_file_open,  
            **kwargs)

    # Cryo card comands
    def get_cryo_card_temp(self):
        """
        Returns:
        --------
        temp (float): Temperature of the cryostat card in Celcius
        """
        return self.C.read_temperature()

    def get_cryo_card_hemt_bias(self):
        """
        Returns:
        --------
        bias (float): The HEMT bias in volts
        """
        return self.C.read_hemt_bias()

    def get_cryo_card_50k_bias(self):
        """
        Returns:
        --------
        bias (float): The 50K bias in volts
        """
        return self.C.read_50k_bias()

    def get_cryo_card_cycle_count(self):
        """
        Returns:
        --------
        cycle_count (float): The cycle count
        """
        self.log('Not doing anything because not implement in cryo_card.py')
        # return self.C.read_cycle_count()

    def get_cryo_card_relays(self):
        """
        Returns:
        --------
        relays (hex): The cryo card relays value
        """
        return self.C.read_relays()

    def set_cryo_card_relays(self, relay, write_log=False):
        """
        Sets the cryo card relays

        Args:
        -----
        relays (hex): The cryo card relays
        """
        if write_log:
            self.log('Writing relay using cryo_card object. {}'.format(relay))
        self.C.write_relays(relay)

    def set_cryo_card_delatch_bit(self, bit, write_log=False):
        """
        Delatches the cryo card for a bit.

        Args:
        -----
        bit (int): The bit to temporarily delatch
        """
        if write_log:
            self.log('Setting delatch bit using cryo_card ' +
                'object. {}'.format(bit))
        self.C.delatch_bit(bit)

    _smurf_to_gcp_stream = 'userConfig[0]'  # bit for streaming
    def get_user_config0(self, as_binary=False, **kwargs):
        """
        """
        val =  self._caget(self.timing_header + 
                           self._smurf_to_gcp_stream, **kwargs)
        if as_binary:
            val = bin(val)

        return val

    def set_smurf_to_gcp_stream(self, val, **kwargs):
        """
        Turns on or off streaming from smurf to GCP.
        This only accepts bools. Annoyingly the bit is 
        0 for streaming and 1 for off. This function takes
        care of that, so True for streaming and False 
        for off.
        """
        old_val = self.get_user_config0()
        if val == False:
            new_val = old_val | (1 << 1)
        elif val == True:
            new_val = old_val
            if old_val & 1 << 1 != 0:
                new_val = old_val & ~(1 << 1)
        self._caput(self.timing_header + 
                    self._smurf_to_gcp_stream, new_val, **kwargs)

    def get_smurf_to_gcp_stream(self, **kwargs):
        """
        """
        return self._caget(self.timing_header + 
                    self._smurf_to_gcp_stream, **kwargs)

    def set_smurf_to_gcp_clear(self, val, **kwargs):
        """
        Clears the wrap counter and average if set to 1.
        Holds it clear until set back to 0.
        """
        clear_bit = 0
        old_val = self.get_user_config0()
        if val:
            new_val = old_val | (1 << clear_bit)
        elif ~val:
            new_val = old_val
            if old_val & 1 << clear_bit != 0:
                new_val = old_val & ~(1 << clear_bit)

        self._caput(self.timing_header +
                    self._smurf_to_gcp_stream, new_val, **kwargs)
        

    _num_rows = 'userConfig[2]'  # bit for num_rows
    def set_num_rows(self, val, **kwargs):
        """
        """
        old = self._caget(self.timing_header +
                    self._num_rows)
        new = (old & 0xFFFF0000) + ((val & 0xFFFF))
        self._caput(self.timing_header +
                    self._num_rows, new, **kwargs)

    def set_num_rows_reported(self, val, **kwargs):
        """
        """
        old = self._caget(self.timing_header +
                    self._num_rows)
        new = (old & 0x0000FFFF) + ((val & 0xFFFF) << 16)
        self._caput(self.timing_header +
                    self._num_rows, new, **kwargs)

    _row_len = 'userConfig[4]'
    def set_row_len(self, val, **kwargs):
        """
        """
        old = self._caget(self.timing_header + 
                    self._row_len)
        new = (old & 0xFFFF0000) + ((val & 0xFFFF))
        self._caput(self.timing_header + 
                    self._row_len, new, **kwargs)

    def set_data_rate(self, val, **kwargs):
        """
        """
        old = self._caget(self.timing_header + 
                    self._row_len)
        new = (old & 0x0000FFFF) + ((val & 0xFFFF)<<16)
        self._caput(self.timing_header +
                    self._row_len, new, **kwargs)
