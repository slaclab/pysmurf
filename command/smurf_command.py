import numpy as np
import os
import epics
import time
from pysmurf.base import SmurfBase

class SmurfCommandMixin(SmurfBase):

    def _caput(self, cmd, val, write_log=False, execute=True, **kwargs):
        '''
        Wrapper around pyrogue lcaput. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        write_log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        '''
        if write_log:
            self.log('caput ' + cmd + ' ' + str(val), self.LOG_USER)

        if execute:
            epics.caput(cmd, val)

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
            if log:
                self.log(ret)
            return ret
        else:
            return None

    def set_defaults_pv(self, **kwargs):
        '''
        '''
        self._caput(self.epics_root + ':AMCc:setDefaults', 1, **kwargs)
        self.log('Setting defaults. Waiting 5 seconds', self.LOG_INFO)
        time.sleep(5)  # This is done in our original script. Not sure why.
        self.log('Done waiting 5 seconds. Defaults are set.', self.LOG_INFO)

    _amplitude_scale_array = 'CryoChannels:amplitudeScaleArray'
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
        return self._caget(self._band_root(band) + self._amplitude_scale_array, 
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
        return self._caget(self._band_root(band) + self._amplitude_scale_array, 
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
        self._caput(self.jesd_root + 
            self._data_out_mux.format(b), val, **kwargs)

    def get_data_out_mux(self, b, **kwargs):
        '''
        '''
        return self._caget(self.jesd_root + self._data_out_mux.format(b), val, 
            **kwargs)

    _link_disable = 'LINK_DISABLE'
    def set_jesd_link_disable(self, val, **kwargs):
        '''
        Disables jesd link
        '''
        self._caput(self.jesd_root_rx + self._link_disable, val, **kwargs)

    def get_jesd_link_disable(self, **kwargs):
        '''
        Disables jesd link
        '''
        return self._caget(self.jesd_root_rx + self._link_disable, val, 
            **kwargs)

    # rtm commands
    _reset_rtm = 'resetRtm'
    def reset_rtm(self, **kwargs):
        '''
        Resets the rear transition module (RTM)
        '''
        self._caput(self.rtm_cryo_det_root + self._reset_rtm, 1, **kwargs)













