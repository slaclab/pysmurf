import numpy as np
import os
import epics
from pysmurf.base import SmurfBase

class SmurfCommandMixin(SmurfBase):

    def _caput(self, cmd, val, log=False, execute=True):
        '''
        Wrapper around pyrogue lcaput. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        '''
        if log:
            self.log('caget ' + cmd + val, self.LOG_USER)

        if execute:
            epics.caput(cmd, val)

    def _caget(self, cmd, log=False, execute=True):
        '''
        Wrapper around pyrogue lcaget. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.

        Returns:
        --------
        ret : The requested value
        '''
        if log:
            self.log('caput ' + cmd, self.LOG_USER)

        if execute:
            ret = epics.caget(cmd)
            if log:
                self.log(ret)
            return ret
        else:
            return None

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
        return self._caget(self._band_root(band) + 
            ':CryoChannels:amplitudeScaleArray', **kwargs)

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
        return self._caget(self._band_root(band) + 
            'CryoChannels:feedbackEnableArray', **kwargs)


    def set_single_channel_readout(self, band, val, **kwargs):
        '''
        Sets the singleChannelReadout bit.

        Args:
        -----
        band (int): The band to set to single channel readout
        '''
        self._caput(self._band_root(band) + 'singleChannelReadout', val, **kwargs)

    def get_single_channel_readout(self, band):
        '''

        '''
        return self._caget(self._band_root(band) + 'singleChannelReadout', **kwargs)

    def set_single_channel_readout2(self, band, val, **kwargs):
        '''
        Sets the singleChannelReadout2 bit.

        Args:
        -----
        band (int): The band to set to single channel readout
        '''
        self._caput(self._band_root(band) + 'singleChannelReadout2', val, **kwargs)


    def get_single_channel_readout2(self, band, **kwargs):
        '''

        '''
        return self._caget(self._band_root(band) + 'singleChannelReadout2', **kwargs)


    def set_iq_stream_enable(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + 'iqStreamEnable', val, **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band)  + 'Base[{}]:iqStreamEnable', 
            **kwargs)

    def set_decimation(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + 'decimation', val, **kwargs)

    def get_decimation(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + 'decimation', **kwargs)

    def set_filterAlpha(self, band, val, **kwargs):
        '''
        '''
        self._caput(self._band_root(band) + 'filterAlpha', val, **kwargs)

    def get_filterAlpha(self, band, **kwargs):
        '''
        '''
        return self._caget(self._band_root(band) + 'Base[{}]:filterAlpha', **kwargs)


