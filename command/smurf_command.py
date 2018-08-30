import numpy as np
import os

class SmurfCommandMixin():

    def _caput(self, cmd, log=False, execute=True):
        '''
        Wrapper around pyrogue lcaput. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        '''
        if log:
            self.log('caget ' + cmd, self.LOG_USER)

        if execute:
            print('This needs to be implemented with pyrogue')

    def _caget(self, cmd, log=False, execute=True):
        '''
        Wrapper around pyrogue lcaget. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        '''
        if log:
            self.log('caput ' + cmd, self.LOG_USER)

        if execute:
            print('This needs to be implemented with pyrogue')
            # write the value if log is True

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
        return self._caget(self.sysgencryo + 
            'Base[{}]:CryoChannels:amplitudeScaleArray'.format(int(band)),
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
        return self_caget(self.sysgencryo + 
            'Base[{}]:CryoChannels:feedbackEnableArray'.format(int(band)),
            **kwargs)        
