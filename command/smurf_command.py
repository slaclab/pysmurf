import numpy as np
import os
import epics
import time
from pysmurf.base import SmurfBase

class SmurfCommandMixin(SmurfBase):

    def _caput(self, cmd, val, write_log=False, execute=True, wait_before=None,
        wait_after=None, **kwargs):
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
        '''
        if wait_before is not None:
            if write_log:
                self.log('Waiting {:3.2f} seconds before...'.format(wait_before),
                    self.LOG_USER)
            time.sleep(wait_before)
        if write_log:
            self.log('caput ' + cmd + ' ' + str(val), self.LOG_USER)

        if execute:
            epics.caput(cmd, val)

        if wait_after is not None:
            if write_log:
                self.log('Waiting {:3.2f} seconds after...'.format(wait_after),
                    self.LOG_USER)
            time.sleep(wait_after)
            self.log('Done waiting.', self.LOG_USER)

    def _caget(self, cmd, write_log=False, execute=True, **kwargs):
        '''
        Wrapper around pyrogue lcaget. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        write_log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.

        Returns:
        --------
        ret : The requested value
        '''
        if write_log:
            self.log('caput ' + cmd, self.LOG_USER)

        if execute:
            ret = epics.caget(cmd)
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
        return self._caget(self._band_root(band) + self._number_sub_bands, band, 
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
        return self._caget(self._band_root(band) + self._number_channels, band,
            **kwargs)

    def set_defaults_pv(self, **kwargs):
        '''
        '''
        self._caput(self.epics_root + ':AMCc:setDefaults', 1, wait_after=5,
            **kwargs)
        # This sleep is in the original script. Not sure why.
        self.log('Defaults are set.', self.LOG_INFO)

    _eta_scan_freqs = 'etaScanFreqs'
    def set_eta_scan_freq(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_freqs, val, 
            **kwargs)

    def get_eta_scan_freq(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_freqs, 
            **kwargs)

    _eta_scan_amplitude = 'etaScanAmplitude'
    def set_eta_scan_amplitude(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_amplitude, val, 
            **kwargs)

    def get_eta_scan_amplitude(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_amplitude, 
            **kwargs)

    _eta_scan_channel = 'etaScanChannel'
    def set_eta_scan_channel(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_channel, val, 
            **kwargs)

    def get_eta_scan_channel(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_channel, 
            **kwargs)

    _eta_scan_dwell = 'etaScanDwell'
    def set_eta_scan_dwell(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_dwell, val, **kwargs)

    def get_eta_scan_dwell(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_dwell, 
            **kwargs)

    _run_eta_scan = 'runEtaScan'
    def set_run_eta_scan(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._cryo_root(band) + self._run_eta_scan, val, **kwargs)

    def get_run_eta_scan(self, band, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._run_eta_scan, **kwargs)    

    _eta_scan_results_real = 'etaScanResultsReal'
    def get_eta_scan_results_real(self, band, count, **kwargs):
        '''
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_results_real,
            count=count, **kwargs)

    _eta_scan_results_imag = 'etaScanResultsImag'
    def get_eta_scan_results_imag(self, band, count, **kwargs):
        '''
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

    def get_single_channel_readout(self, band):
        '''

        '''
        return self._caget(self._band_root(band) + self._single_channel_readout, 
            **kwargs)

    _single_channel_readout2 = 'singleChannelReadout2'
    def set_single_channel_readout2(self, band, val, **kwargs):
        '''
        Sets the singleChannelReadout2 bit.

        Args:
        -----
        band (int): The band to set to single channel readout
        '''
        self._caput(self._band_root(band) + self._single_channel_readout2, val, 
            **kwargs)

    def get_single_channel_readout2(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._single_channel_readout2, 
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
        '''
        self._caput(self._band_root(band) + self._filter_alpha, val, **kwargs)

    def get_filter_alpha(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._filter_alpha, **kwargs)

    _iq_swap_in = 'iqSwapIn'
    def set_iq_swap_in(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._iq_swap_in, val, **kwargs)

    def get_iq_swap_in(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._iq_swap_in, **kwargs)

    _iq_swap_out = 'iqSwapOut'
    def set_iq_swap_out(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._iq_swap_out, val, **kwargs)

    def get_iq_swap_out(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._iq_swap_out, **kwargs)

    _ref_phase_delay = 'refPhaseDelay'
    def set_ref_phase_delay(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._ref_phase_delay, val, 
            **kwargs)

    def get_ref_phase_delay(self, band, **kwargs):
        '''
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
        '''
        self._caput(self._band_root(band) + self._tone_scale, val, **kwargs)

    def get_tone_scale(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._tone_scale, **kwargs)

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

    _lms_gain = 'lmsGain'
    def set_lms_gain(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + self._lms_gain, val, **kwargs)

    def get_lms_gain(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + self._lms_gain, **kwargs)

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
        return self._caget(self._band_root(band) + self._digitizer_frequency_mhz, 
            **kwargs)

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
        return self._caget(self.jesd_tx_root + self._data_out_mux.format(b), val, 
            **kwargs)

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

    # rtm commands
    _reset_rtm = 'resetRtm'
    def reset_rtm(self, **kwargs):
        '''
        Resets the rear transition module (RTM)
        '''
        self._caput(self.rtm_cryo_det_root + self._reset_rtm, 1, **kwargs)

    _cfg_reg_ena_bit = 'CfgRegEnaBit'
    def set_cfg_reg_ena_bit(self, val, **kwargs):
        '''
        '''
        self._caput(self.rtm_spi_root + self._cfg_reg_ena_bit, val, **kwargs)

    def get_cfg_reg_ena_bit(self, **kwargs):
        '''
        '''
        return self._caget(self.rtm_spi_root + self._cfg_reg_ena_bit, **kwargs)

    def flux_ramp_on(self, **kwargs):
        '''
        Turns on the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        '''
        self.set_reg_ena_bit(1, **kwargs)

    def flux_ramp_off(self, **kwargs):
        '''
        Turns off the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        '''
        self.set_reg_ena_bit(0, **kwargs)

    _hemt_v = 'HemtBiasDacCtrlRegCh[33]'
    def set_hemt_v(self, val, override=False, **kwargs):
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

    def get_hemt_v(self, **kwargs):
        '''
        Returns the HEMT voltage in bits.
        '''
        return self._caget(self.rtm_spi_max_root + self._hemt_v, **kwargs)





