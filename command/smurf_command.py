import numpy as np
import os
import epics
import time
from pysmurf.base import SmurfBase
from pysmurf.command.sync_group import SyncGroup as SyncGroup

class SmurfCommandMixin(SmurfBase):


    _global_poll_enable = ':AMCc:enable'
    def _caput(self, cmd, val, write_log=False, execute=True, wait_before=None,
        wait_after=None, wait_done=True, log_level=0, enable_poll=False,
        disable_poll=False, new_epics_root=None):
        '''
        Wrapper around pyrogue lcaput. Puts variables into epics

        Args:
        -----
        cmd : The pyrogue command to be executed. Input as a string
        val: The value to put into epics

        Optional Args:
        --------------
        write_log (bool) : Whether to log the data or not. Default False
        log_level (int):
        execute (bool) : Whether to actually execute the command. Defualt True.
        wait_before (int) : If not None, the number of seconds to wait before
            issuing the command
        wait_after (int) : If not None, the number of seconds to wait after
            issuing the command
        wait_done (bool) : Wait for the command to be finished before returning.
            Default True.
        enable_poll (bool) : Allows requests of all PVs. Default False.
        disable_poll (bool) : Disables requests of all PVs after issueing command.
            Default False.
        new_epics_root (str) : Temporarily replaces current epics root with
            a new one.
        '''
        if new_epics_root is not None:
            self.log('Temporarily using new epics root: {}'.format(new_epics_root))
            old_epics_root = self.epics_root
            self.epics_root = new_epics_root
            cmd = cmd.replace(old_epics_root, self.epics_root)
            
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

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
            if write_log:
                self.log('Done waiting.', self.LOG_USER)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        if new_epics_root is not None:
            self.epics_root = old_epics_root
            self.log('Returning back to original epics root'+
                     ' : {}'.format(self.epics_root))
        
    def _caget(self, cmd, write_log=False, execute=True, count=None,
               log_level=0, enable_poll=False, disable_poll=False,
               new_epics_root=None):
        '''
        Wrapper around pyrogue lcaget. Gets variables from epics

        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string

        Opt Args:
        ---------
        write_log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        count (int or None) : number of elements to return for array data
        enable_poll (bool) : Allows requests of all PVs. Default False.
        disable_poll (bool) : Disables requests of all PVs after issueing command.
            Default False.
        new_epics_root (str) : Temporarily replaces current epics root with
            a new one.

        Returns:
        --------
        ret : The requested value
        '''
        if new_epics_root is not None:
            self.log('Temporarily using new epics root: {}'.format(new_epics_root))
            old_epics_root = self.epics_root
            self.epics_root = new_epics_root
            cmd = cmd.replace(old_epics_root, self.epics_root)
            
        if enable_poll:
            epics.caput(self.epics_root+ self._global_poll_enable, True)

        if write_log:
            self.log('caget ' + cmd, log_level)

        if execute and not self.offline:
            ret = epics.caget(cmd, count=count)
            if write_log:
                self.log(ret)
        else:
            ret = None

        if disable_poll:
            epics.caput(self.epics_root+ self._global_poll_enable, False)

        if new_epics_root is not None:
            self.epics_root = old_epics_root
            self.log('Returning back to original epics root'+
                     ' : {}'.format(self.epics_root))

        return ret

    def get_enable(self, **kwargs):
        """
        Returns the status of the global poll bit epics_root:AMCc:enable.
        If False, pyrogue is not currently polling the server. PVs will
        not be updating.
        """
        return self._caget(self.epics_root + self._global_poll_enable,
                           enable_poll=False, disable_poll=False, **kwargs)


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


    def run_pwr_up_sys_ref(self,bay, **kwargs):
        """
        """
        triggerPV=self.lmk.format(bay) + 'PwrUpSysRef'
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log('{} sent'.format(triggerPV), self.LOG_USER)

    _eta_scan_in_progress = 'etaScanInProgress'
    def get_eta_scan_in_progress(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_in_progress,
                    **kwargs)

    _gradient_descent_max_iters = 'gradientDescentMaxIters'
    def set_gradient_descent_max_iters(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_max_iters, val,
                    **kwargs)

    def get_gradient_descent_max_iters(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_max_iters,
                           **kwargs)


    _gradient_descent_averages = 'gradientDescentAverages'
    def set_gradient_descent_averages(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_averages, val,
                    **kwargs)

    def get_gradient_descent_averages(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_averages,
                           **kwargs)

    _gradient_descent_gain = 'gradientDescentGain'
    def set_gradient_descent_gain(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_gain, val,
                    **kwargs)

    def get_gradient_descent_gain(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_gain,
                           **kwargs)


    _gradient_descent_converge_khz = 'gradientDescentConvergeHz'
    def set_gradient_descent_converge_hz(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_converge_khz, val,
                    **kwargs)

    def get_gradient_descent_converge_hz(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_converge_khz,
                           **kwargs)

    _gradient_descent_step_khz = 'gradientDescentStepHz'
    def set_gradient_descent_step_hz(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_step_khz, val,
                    **kwargs)

    def get_gradient_descent_step_hz(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_step_khz,
                           **kwargs)


    _gradient_descent_momentum = 'gradientDescentMomentum'
    def set_gradient_descent_momentum(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_momentum, val,
                    **kwargs)

    def get_gradient_descent_momentum(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_momentum,
                           **kwargs)

    _gradient_descent_beta = 'gradientDescentBeta'
    def set_gradient_descent_beta(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_beta, val,
                    **kwargs)

    def get_gradient_descent_beta(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_beta,
                           **kwargs)


    def run_parallel_eta_scan(self, band, sync_group=True, **kwargs):
        """
        runParallelScan
        """
        triggerPV=self._cryo_root(band) + 'runParallelEtaScan'
        monitorPV=self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log('{} sent'.format(triggerPV), self.LOG_USER)

        if sync_group:
            sg = SyncGroup([monitorPV])
            sg.wait()
            vals = sg.get_values()
            self.log('parallel etaScan complete ; etaScanInProgress = {}'.format(vals[monitorPV]),
                     self.LOG_USER)

    _run_serial_eta_scan = 'runSerialEtaScan'
    def run_serial_eta_scan(self, band, sync_group=True, timeout=240,
                            **kwargs):
        """
        Does an eta scan serially across the entire band. You must
        already be tuned close to the resontor dip. Use
        run_serial_gradient_descent to get it.

        Args:
        ----_
        band (int) : The band to eta scan

        Opt Args:
        --------
        sync_group (bool) : Whether to use the sync group to monitor
            the PV. Defauult is True.
        timeout (float) : The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_eta_scan
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            vals = sg.get_values()


    _run_serial_min_search = 'runSerialMinSearch'
    def run_serial_min_search(self, band, sync_group=True, timeout=240,
                              **kwargs):
        """
        Does a brute force search for the resonator minima. Starts at
        the currently set frequency.

        Args:
        ----
        band (int) : The band the min search

        Opt Args:
        --------
        sync_group (bool) : Whether to use the sync group to monitor
            the PV. Defauult is True.
        timeout (float) : The maximum amount of time to wait for the PV.
        """
        triggerPV = self._cryo_root(band) + self._run_serial_min_search
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            vals = sg.get_values()


    _run_serial_gradient_descent = 'runSerialGradientDescent'
    def run_serial_gradient_descent(self, band, sync_group=True,
                                    timeout=240, **kwargs):
        """
        Does a gradient descent search for the minimum.


        Args:
        ----
        band (int) : The band the min search

        Opt Args:
        --------
        sync_group (bool) : Whether to use the sync group to monitor
            the PV. Defauult is True.
        timeout (float) : The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_gradient_descent
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            vals = sg.get_values()


    _selextref = "SelExtRef"
    def sel_ext_ref(self, bay, **kwargs):
        """
        Selects this bay to trigger off of external reference (through front panel)

        Args:
        ----
        bay (int) : which bay to set to ext ref.  Either 0 or 1.
        """
        assert (bay in [0,1]),'bay must be an integer and in [0,1]'
        triggerPV=self.microwave_mux_core.format(bay) + self._selextref
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log('{} sent'.format(triggerPV), self.LOG_USER)


    _writeconfig = ":AMCc:WriteConfig"
    def write_config(self, val, **kwargs):
        """
        Writes the current PyRogue settings to a yml file.

        Args:
        ----
        val (str) : The path (including file name) to write the yml file to.
        """
        self._caput(self.epics_root + self._writeconfig,
                    val, **kwargs)


    _tone_file_path = 'CsvFilePath'
    def get_tone_file_path(self, bay, **kwargs):
        """
        Returns the tone file path that's currently being used for this bay.

        Args:
        ----
        bay (int) : Which AMC bay (0 or 1).
        """

        ret=self._caget(self.dac_sig_gen.format(bay) + self._tone_file_path,**kwargs)
        # convert to human-readable string
        ret=''.join([str(s, encoding='UTF-8') for s in ret])
        # strip \x00 - not sure why I have to do this
        ret=ret.strip('\x00')
        return ret

    def set_tone_file_path(self, bay, val, **kwargs):
        """
        Sets the tone file path for this bay.

        Args:
        ----
        bay (int) : Which AMC bay (0 or 1).
        val (str) : Path (including csv file name) to tone file.
        """
        # make sure file exists before setting

        if not os.path.exists(val):
            self.log('Tone file {} does not exist!  Doing nothing!'.format(val),
                     self.LOG_ERROR)
            raise ValueError('Must provide a path to an existing tone file.')
        
        self._caput(self.dac_sig_gen.format(bay) + self._tone_file_path,
                    val, **kwargs)

    _load_tone_file = 'LoadCsvFile'        
    def load_tone_file(self, bay, val=None, **kwargs):
        """
        Loads tone file specified in tone_file_path.

        Args:
        ----
        bay (int) : Which AMC bay (0 or 1).

        Optional Args:
        --------------
        val (str) : Path (including csv file name) to tone file.  If
                    none provided, assumes something valid has already
                    been loaded into DacSigGen[#]:CsvFilePath
        """

        # Set tone file path if provided.
        if val is not None:
            self.set_tone_file_path(bay, val)
        else:
            val=self.get_tone_file_path(bay)


        self.log('Loading tone file : {}'.format(val),
                 self.LOG_USER)        
        self._caput(self.dac_sig_gen.format(bay) + self._load_tone_file, val, **kwargs)        

    _tune_file_path = 'tuneFilePath'
    def set_tune_file_path(self, val, **kwargs):
        """
        """
        self._caput(self.sysgencryo + self._tune_file_path,
                    val, **kwargs)

    def get_tune_file_path(self, **kwargs):
        """
        """
        return self._caget(self.sysgencryo + self._tune_file_path,
                           **kwargs)

    _load_tune_file = 'loadTuneFile'
    def set_load_tune_file(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._load_tune_file,
                    val, **kwargs)

    def get_load_tune_file(self, band, **kwargs):
        """
        """
        self._caget(self._cryo_root(band) + self._load_tune_file,
                    **kwargs)


    _eta_scan_del_f = 'etaScanDelF'
    def set_eta_scan_del_f(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._eta_scan_del_f, val,
                    **kwargs)

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

    _eta_scan_averages = 'etaScanAverages'
    def set_eta_scan_averages(self, band, val, **kwargs):
        '''
        Sets the number of frequency error averages to take at each point of the etaScan.

        Args:
        -----
        band (int) : The band to set
        val (int) : The channel to set
        '''
        self._caput(self._cryo_root(band) + self._eta_scan_averages, val,
            **kwargs)

    def get_eta_scan_averages(self, band, **kwargs):
        '''
        Gets the number of frequency error averages taken at each point of the etaScan.

        Args:
        -----
        band (int) : The band to set

        Returns:
        --------
        eta_scan_averages (int) : The number of frequency error averages taken at each point of the etaScan.
        '''
        return self._caget(self._cryo_root(band) + self._eta_scan_averages,
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

    _stream_enable = 'enableStreaming'
    def set_stream_enable(self, val, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        self._caput(self.app_core + self._stream_enable, val, **kwargs)

    def get_stream_enable(self, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        return self._caget(self.app_core + self._stream_enable,
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

    _loop_filter_output_array = 'loopFilterOutputArray'
    def get_loop_filter_output_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._loop_filter_output_array,
            **kwargs)

    _tone_frequency_offset_mhz = 'toneFrequencyOffsetMHz'
    #def get_tone_frequency_offset_mhz(self, band, **kwargs):
    #    """
    #    """
    #    return self._caget(self._band_root(band) + self._tone_frequency_offset_mhz,
    #        **kwargs)

    def get_tone_frequency_offset_mhz(self, **kwargs):
        def bitRevOrder(x):
            length = x.shape[0]
            assert( np.log2(length) == np.floor(np.log2(length)) )

            if length == 1:
                return x
            else:
                evenIndex = np.arange(length/2)*2
                oddIndex  = np.arange(length/2)*2 + 1
                evens     = bitRevOrder( x[evenIndex.astype(int)] )
                odds      = bitRevOrder( x[oddIndex.astype(int)] )
                return np.concatenate([evens, odds])

        fAdc          = 614.4      # digitizer frequency MHz
        fftLen        = 64         # fft length
        nFft          = 2          # interleaved polyphase filter bank
        nTones        = 4          # 4 tones/subband
        bands         = np.arange(64) * fAdc/fftLen
        bandsRevOrder = bands[ bitRevOrder( np.arange(64) ) ]
        bandsTile1    = np.tile(bandsRevOrder, nTones) 
        bandsTile2    = np.concatenate([bandsTile1, bandsTile1 + fAdc/(fftLen*nFft)])
        
        bandsCenter   = bandsTile2
        index         = np.where( bandsCenter > fAdc/2 )
        bandsCenter[index]  = bandsCenter[index] - fAdc
        return bandsCenter
    
    def	set_tone_frequency_offset_mhz(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) +
                    self._tone_frequency_offset_mhz, val,
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

    _frequency_error_array = 'frequencyErrorArray'
    def set_frequency_error_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._frequency_error_array, val,
            **kwargs)

    def get_frequency_error_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._frequency_error_array,
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

    _trigger_reset_delay = 'trigRstDly'
    def set_trigger_reset_delay(self, band, val, **kwargs):
        '''
        Trigger reset delay, set such that the ramp resets at the flux ramp glitch.  2.4 MHz ticks.

        Args:
        -----
        band (int) : Which band.
        '''
        self._caput(self._band_root(band) + self._trigger_reset_delay, val, **kwargs)

    def get_trigger_reset_delay(self, band, **kwargs):
        '''
        Trigger reset delay, set such that the ramp resets at the flux ramp glitch.  2.4 MHz ticks.

        Args:
        -----
        band (int) : Which band.
        '''
        return self._caget(self._band_root(band) + self._trigger_reset_delay, **kwargs)

    _feedback_start = 'feedbackStart'
    def set_feedback_start(self, band, val, **kwargs):
        '''
	The flux ramp DAC value at which to start applying feedback in each flux ramp cycle.
	In 2.4 MHz ticks.

        Args:
        -----
        band (int) : Which band.
        '''
        self._caput(self._band_root(band) + self._feedback_start, val, **kwargs)

    def get_feedback_start(self, band, **kwargs):
        '''
	The flux ramp DAC value at which to start applying feedback in each flux ramp cycle.
	In 2.4 MHz ticks.	

        Args:
        -----
        band (int) : Which band.
        '''
        return self._caget(self._band_root(band) + self._feedback_start, **kwargs)

    _feedback_end = 'feedbackEnd'
    def set_feedback_end(self, band, val, **kwargs):
        '''
	The flux ramp DAC value at which to stop applying feedback in each flux ramp cycle.
	In 2.4 MHz ticks.

        Args:
        -----
        band (int) : Which band.
        '''
        self._caput(self._band_root(band) + self._feedback_end, val, **kwargs)

    def get_feedback_end(self, band, **kwargs):
        '''
	The flux ramp DAC value at which to stop applying feedback in each flux ramp cycle.
	In 2.4 MHz ticks.	

        Args:
        -----
        band (int) : Which band.
        '''
        return self._caget(self._band_root(band) + self._feedback_end, **kwargs)    

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
        Returns the center frequency of the band in MHz
        '''
        if self.offline:
            if band == 3:
                bc = 5.75E3
            elif band == 2:
                bc = 5.25E3
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
        Returns the digitizer frequency in MHz.
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
        Set the amplitude scale for a single channel. The amplitude scale
        defines the power of the tone.

        Args:
        -----
        band (int): The band the channel is in
        channel (int): The channel number
        val (int): The value of the tone amplitude. Acceptable units are
            0 to 15.
        """
        self._caput(self._channel_root(band, channel) +
            self._amplitude_scale_channel, val, **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
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
        return self._caget(self._channel_root(band, channel) +
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


    def band_to_bay(self,b):
        '''
        Returns the bay index for the band.
        Assumes LB is plugged into bay 0,  corresponding to bands [0,1,2,3] and that
        HB is plugged into bay 1, corresponding to bands [4,5,6,7].

        Args:
        -----
        b (int): Band number.
        '''
        if b in [0,1,2,3]:
            bay=0
        elif b in [4,5,6,7]:
            bay=1
        else:
            assert False, 'band supplied to band_to_bay() must be and integer in [0,1,2,3,4,5,6,7] ...'
        return bay

    # Attenuator
    _uc = 'UC[{}]'
    def set_att_uc(self, b, val, **kwargs):
        '''
        Set the upconverter attenuator

        Args:
        -----
        b (int): The band number.
        val (int): The attenuator value
        '''
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        self._caput(self.att_root.format(bay) + self._uc.format(int(att)), val, **kwargs)

    def get_att_uc(self, b, **kwargs):
        '''
        Get the upconverter attenuator value

        Args:
        -----
        b (int): The band number.
        '''
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        return self._caget(self.att_root.format(bay) + self._uc.format(int(att)), **kwargs)


    _dc = 'DC[{}]'
    def set_att_dc(self, b, val, **kwargs):
        '''
        Set the down-converter attenuator

        Args:
        -----
        b (int): The band number.
        val (int): The attenuator value
        '''
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        self._caput(self.att_root.format(bay) + self._dc.format(int(att)), val, **kwargs)

    def get_att_dc(self, b, **kwargs):
        '''
        Get the down-converter attenuator value

        Args:
        -----
        b (int): The band number.
        '''
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        return self._caget(self.att_root.format(bay) + self._dc.format(int(att)), **kwargs)

    # ADC commands
    _adc_remap = "Remap[0]"  # Why is this hardcoded 0
    def set_remap(self, **kwargs):
        '''
        This command should probably be renamed to something more descriptive.
        '''
        self._caput(self.adc_root + self._adc_remap, 1, **kwargs)

    # DAC commands
    _dac_temp = "Temperature"
    def get_dac_temp(self, bay, dac, **kwargs):
        '''
        Get temperature of the DAC in celsius

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        '''
        return self._caget(self.dac_root.format(bay,dac) + self._dac_temp, **kwargs)

    _dac_enable = "enable"
    def set_dac_enable(self, bay, dac, val, **kwargs):
        '''
        Enables DAC

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        val (int): Value to set the DAC enable register to [0 or 1].
        '''
        self._caput(self.dac_root.format(bay,dac) + self._dac_enable, val, **kwargs)

    def get_dac_enable(self, bay, dac, **kwargs):
        '''
        Gets enable status of DAC

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        '''
        return self._caget(self.dac_root.format(bay,dac) + self._dac_enable, **kwargs)

    # Jesd commands
    _data_out_mux = 'dataOutMux[{}]'
    def set_data_out_mux(self, bay, b, val, **kwargs):
        '''
        '''
        self._caput(self.jesd_tx_root.format(bay) +
            self._data_out_mux.format(b), val, **kwargs)

    def get_data_out_mux(self, bay, b, **kwargs):
        '''
        '''
        return self._caget(self.jesd_tx_root.format(bay) + self._data_out_mux.format(b),
            val, **kwargs)

    # Jesd DAC commands
    _jesd_reset_n = "JesdRstN"
    def set_jesd_reset_n(self, bay, dac, val, **kwargs):
        '''
        Set DAC JesdRstN

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        val (int): Value to set JesdRstN to [0 or 1].
        '''
        self._caput(self.dac_root.format(bay,dac) + self._jesd_reset_n, val, **kwargs)

    _jesd_rx_enable = 'Enable'
    def get_jesd_rx_enable(self, bay, **kwargs):
        return self._caget(self.jesd_rx_root.format(bay) + self._jesd_rx_enable, **kwargs)

    def set_jesd_rx_enable(self, bay, val, **kwargs):
        self._caput(self.jesd_rx_root.format(bay) + self._jesd_rx_enable, val, **kwargs)

    _jesd_rx_data_valid = 'DataValid'
    def get_jesd_rx_data_valid(self, bay, **kwargs):
        return self._caget(self.jesd_rx_root.format(bay) + self._jesd_rx_data_valid, **kwargs)

    _link_disable = 'LINK_DISABLE'
    def set_jesd_link_disable(self, bay, val, **kwargs):
        '''
        Disables jesd link
        '''
        self._caput(self.jesd_rx_root.format(bay) + self._link_disable, val, **kwargs)

    def get_jesd_link_disable(self, bay, **kwargs):
        '''
        Disables jesd link
        '''
        return self._caget(self.jesd_rx_root.format(bay) + self._link_disable, val,
            **kwargs)

    _jesd_tx_enable = 'Enable'
    def get_jesd_tx_enable(self, bay, **kwargs):
        return self._caget(self.jesd_tx_root.format(bay) + self._jesd_tx_enable, **kwargs)

    def set_jesd_tx_enable(self, bay, val, **kwargs):
        self._caput(self.jesd_tx_root.format(bay) + self._jesd_tx_enable, val, **kwargs)

    _jesd_tx_data_valid = 'DataValid'
    def get_jesd_tx_data_valid(self, bay, **kwargs):
        return self._caget(self.jesd_tx_root.format(bay) + self._jesd_tx_data_valid, **kwargs)

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
        git_hash (str) : The git hash of the FPGA
        '''
        gitHash=self._caget(self.axi_version + self._fpga_git_hash, **kwargs)
        return ''.join([str(s, encoding='UTF-8') for s in gitHash])

    _fpga_git_hash_short = 'GitHashShort'
    def get_fpga_git_hash_short(self, **kwargs):
        '''
        Returns:
        git_hash_short (str) : The short git hash of the FPGA
        '''
        gitHashShort=self._caget(self.axi_version + self._fpga_git_hash_short, **kwargs)
        return ''.join([str(s, encoding='UTF-8') for s in gitHashShort])


    _fpga_build_stamp = 'BuildStamp'
    def get_fpga_build_stamp(self, **kwargs):
        '''
        Returns:
        build_stamp (str) : The FPGA build stamp
        '''
        return self._caget(self.axi_version + self._fpga_build_stamp, **kwargs)

    _input_mux_sel = 'InputMuxSel[{}]'
    def set_input_mux_sel(self, bay, lane, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._input_mux_sel.format(lane), val,
            **kwargs)

    def get_input_mux_sel(self, bay, lane, **kwargs):
        """
        """
        self._caget(self.daq_mux_root.format(bay) + self._input_mux_sel.format(lane),
            **kwargs)

    _data_buffer_size = 'DataBufferSize'
    def set_data_buffer_size(self, bay, val, **kwargs):
        """
        Sets the data buffer size for the DAQx
        """
        self._caput(self.daq_mux_root.format(bay) + self._data_buffer_size, val, **kwargs)

    def get_data_buffer_size(self, bay, **kwargs):
        """
        Gets the data buffer size for the DAQs
        """
        return self._caget(self.daq_mux_root.format(bay) + self._data_buffer_size,
            **kwargs)

    # Waveform engine commands
    _start_addr = 'StartAddr[{}]'
    def set_waveform_start_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.
        val=What value to set.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._start_addr.format(engine), val, **kwargs)

    def get_waveform_start_addr(self, bay, engine, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """

        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
                          self._start_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _end_addr = 'EndAddr[{}]'
    def set_waveform_end_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.
        val=What value to set.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._end_addr.format(engine), val, **kwargs)

    def get_waveform_end_addr(self, bay, engine, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._end_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _wr_addr = 'WrAddr[{}]'
    def set_waveform_wr_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.
        val=What val to set.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr.format(engine), val, **kwargs)

    def get_waveform_wr_addr(self, bay, engine, convert=True, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.

        Optional Arg:
        -------------
        convert (bool) : Convert the output from a string of hex values to an
            int. Default (True)
        """
        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _empty = 'Empty[{}]'
    def set_waveform_empty(self, bay, engine, val, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.
        val=What value to set.
        """
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._empty.format(engine), **kwargs)

    def get_waveform_empty(self, bay, engine, **kwargs):
        """
        bay=Which bay.
        engine=Which waveform engine.
        """
        return self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._empty.format(engine), **kwargs)

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
    def set_trigger_daq(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._trigger_daq, val,
            **kwargs)

    def get_trigger_daq(self, bay, **kwargs):
        """
        """
        self._caget(self.daq_mux_root.format(bay) + self._trigger_daq,
            **kwargs)

    _arm_hw_trigger = "ArmHwTrigger"
    def set_arm_hw_trigger(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._arm_hw_trigger, val, **kwargs)

    _trigger_hw_arm = 'TriggerHwArm'
    def set_trigger_hw_arm(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._trigger_hw_arm, val, **kwargs)

    def get_trigger_hw_arm(self, bay, **kwargs):
        """
        """
        return self._caget(self.daq_mux_root.format(bay) + self._trigger_hw_arm, **kwargs)

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

    _enable_ramp_trigger = 'EnableRampeTrigger'
    def set_enable_ramp_trigger(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._enable_ramp_trigger,
                    vale, **kwargs)

    timing_crate_root = ":AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:EvrV2CoreTriggers"
    _trigger_rate_sel = ":EvrV2ChannelReg[0]:RateSel"
    def set_ramp_rate(self, val, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz

        Allowed rates: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15kHz (hardcoded by timing)
        """
        rate_sel = self.flux_ramp_rate_to_PV(val)

        if rate_sel is not None:
            self._caput(self.epics_root + self.timing_crate_root +
                self._trigger_rate_sel, rate_sel, **kwargs)
        else:
            print("Rate requested is not allowed by timing triggers. Allowed rates are 1, 2, 3, 4, 5, 6, 8, 10, 12, 15kHz only")

    def get_ramp_rate(self, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz
        """

        rate_sel = self._caget(self.epics_root + self.timing_crate_root +
            self._trigger_rate_sel, **kwargs)

        reset_rate = self.flux_ramp_PV_to_rate(rate_sel)

        return reset_rate

    _trigger_delay = ":EvrV2TriggerReg[0]:Delay"
    def set_trigger_delay(self, val, **kwargs):
        """
        Adds an offset to flux ramp trigger.  Only really useful if
        you're using two carriers at once and you're trying to
        synchronize them.  Mitch thinks it's in units of 122.88MHz
        ticks.
        """
        self._caput(self.epics_root + self.timing_crate_root +
                    self._trigger_delay, val, **kwargs)

    def get_trigger_delay(self, **kwargs):
        """
        The flux ramp trigger offset.  Only really useful if you're
        using two carriers at once and you're trying to synchronize
        them.  Mitch thinks it's in units of 122.88MHz ticks.
        """

        trigger_delay = self._caget(self.epics_root + self.timing_crate_root +
                                    self._trigger_delay, **kwargs)
        return trigger_delay

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

    _flux_ramp_dac = 'LTC1668RawDacData'
    def set_flux_ramp_dac(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._flux_ramp_dac, val, **kwargs)

    def get_flux_ramp_dac(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._flux_ramp_dac, **kwargs)

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

    _tes_bias_enable_array = 'TesBiasDacCtrlRegChArray'
    def set_tes_bias_enable_array(self, val, **kwargs):
        """
        Set the TES bias DAC enable bits all at once

        Args:
          val (int array): length 32, addresses the DACs in DAC ordering
        """

        self._caput(self.rtm_spi_max_root + self._tes_bias_enable_array, val,
            **kwargs)

    def get_tes_bias_enable_array(self, **kwargs):
        """
        Get the TES bias DAC enable bits all at once

        Returns a numpy array of size (32,) for each of the DACs
        """

        return self._caget(self.rtm_spi_max_root + self._tes_bias_enable_array,
            **kwargs)

    #_bit_to_V_50k = 2.035/float(2**19)
    #_dac_num_50k = 2
    def set_50k_amp_gate_voltage(self, voltage, override=False, **kwargs):
        """
        """
        if (voltage > 0 or voltage < -1.) and not override:
            self.log('Voltage must be between -1 and 0. Doing nothing.')
        else:
            self.set_tes_bias(self._dac_num_50k, voltage/self._bit_to_V_50k,
                **kwargs)

    def get_50k_amp_gate_voltage(self, **kwargs):
        """
        """
        return self._bit_to_V_50k * self.get_tes_bias(self._dac_num_50k, **kwargs)

    def set_50k_amp_enable(self, disable=False, **kwargs):
        """
        Sets the 50K amp bit to 2 for enable and 0 for disable.

        Opt Args:
        ---------
        disable (bool) : Disable the 50K amplifier. Default False.
        """
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
        val (int) : the TES bias current in DAC units
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

    _tes_bias_array = 'TesBiasDacDataRegChArray'
    def set_tes_bias_array(self, val, **kwargs):
        """
        Set the TES bias DACs. Must give all 32 values.

        Args:
        -----
        val (int array): TES biases to set for each DAC. Expects np array of size
          (32,) in DAC units.
        """

        val[np.ravel(np.where(val > 2**19-1))] = 2**19-1
        if len(np.ravel(np.where(val > 2**19-1))) > 0:
            self.log('Bias too high for some values. Must be <= 2^19 -1. Setting to ' +
            'max value', self.LOG_ERROR)

        val[np.ravel(np.where(val < - 2**19))] = -2**19
        if len(np.ravel(np.where(val < - 2**19))) > 0:
            self.log('Bias too low for some values. Must be >= -2^19. Setting to ' +
            'min value', self.LOG_ERROR)
        self._caput(self.rtm_spi_max_root + self._tes_bias_array, val, **kwargs)

    def get_tes_bias_array(self, **kwargs):
        """
        Get the TES bias for all 32 DACs. Returns in DAC units.

        Returns:
        -----
        bias_array (int array): Size (32,) array of DAC values, in DAC units
        """

        return self._caget(self.rtm_spi_max_root + self._tes_bias_array,
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

    def set_tes_bias_array_volt(self, val, **kwargs):
        """
        Set TES bias DACs. Must give 32 values. Converted to volts based on
          DAC full scale.

        Args:
        -----
        val (float array): TES biases to set for each DAC. Expects np array
          of size (32,) in volts.
        """
        int_val = np.array(val / self._bit_to_volt, dtype=int)

        self.set_tes_bias_array(int_val, **kwargs)

    def get_tes_bias_array_volt(self, **kwargs):
        """
        Get TES bias DAC settings in volt units.

        Returns:
        -----
        bias_array (float array): Size (32,) array of DAC values in volts
        """
        return self._bit_to_volt * self.get_tes_bias_array(**kwargs)

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
    def set_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._low_cycle, val, **kwargs)

    def get_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(self.rtm_cryo_det_root + self._low_cycle, **kwargs)

    _high_cycle = 'HighCycle'
    def set_high_cycle(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).
        Along with LowCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._high_cycle, val, **kwargs)

    def get_high_cycle(self, val, **kwargs):
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

    _enable_ramp = 'EnableRamp'
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
	0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_start_mode, val,
            **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        """
        Select Ramp to the CPLD
	0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
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

    # can't write a get for this right now because read back isn't implemented
    # I think...
    _hemt_v_enable = 'HemtBiasDacCtrlRegCh[33]'
    def set_hemt_enable(self, disable=False, zero_gate=False, **kwargs):
        """
        Sets bit to 2 for enable and 0 for disable.

        Opt Args:
        ---------
        disable (bool): If True, sets the bit to 0.
        """
        if disable:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 0,
                **kwargs)
        else:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 2,
                **kwargs)

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
        self.set_hemt_enable()
        if (voltage > self._hemt_gate_max_voltage or voltage < self._hemt_gate_min_voltage ) and not override:
            self.log('Input voltage too high. Not doing anything.' +
                ' If you really want it higher, use the override optional arg.')
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
        return self._bit_to_V_hemt*(self.get_hemt_bias(**kwargs))


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

    # UltraScale+ FPGA
    fpga_root = ":AMCc:FpgaTopLevel:AmcCarrierCore:AxiSysMonUltraScale"
    _fpga_temperature = ":Temperature"
    def get_fpga_temp(self, **kwargs):
        """
        Gets the temperature of the UltraScale+ FPGA.  Returns float32,
        the temperature in degrees Celsius.

        Returns:
        --------
        val (float): The UltraScale+ FPGA temperature in degrees Celsius.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_temperature, **kwargs)

    _fpga_vccint = ":VccInt"
    def get_fpga_vccint(self, **kwargs):
        """
        Returns:
        --------
        val (float): The UltraScale+ FPGA VccInt in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccint, **kwargs)

    _fpga_vccaux = ":VccAux"
    def get_fpga_vccaux(self, **kwargs):
        """
        Returns:
        --------
        val (float): The UltraScale+ FPGA VccAux in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccaux, **kwargs)

    _fpga_vccbram = ":VccBram"
    def get_fpga_vccbram(self, **kwargs):
        """
        Returns:
        --------
        val (float): The UltraScale+ FPGA VccBram in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccbram, **kwargs)

    # Cryo card comands
    def get_cryo_card_temp(self, enable_poll=False, disable_poll=False):
        """
        Returns:
        --------
        temp (float): Temperature of the cryostat card in Celsius
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        T = self.C.read_temperature()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return T


    def get_cryo_card_hemt_bias(self, enable_poll=False, disable_poll=False):
        """
        Returns:
        --------
        bias (float): The HEMT bias in volts
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        hemt_bias = self.C.read_hemt_bias()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return hemt_bias

    def get_cryo_card_50k_bias(self, enable_poll=False, disable_poll=False):
        """
        Returns:
        --------
        bias (float): The 50K bias in volts
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        bias = self.C.read_50k_bias()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return bias

    def get_cryo_card_cycle_count(self, enable_poll=False, disable_poll=False):
        """
        Returns:
        --------
        cycle_count (float): The cycle count
        """
        self.log('Not doing anything because not implement in cryo_card.py')
        # return self.C.read_cycle_count()

    def get_cryo_card_relays(self, enable_poll=False, disable_poll=False):
        """
        Returns:
        --------
        relays (hex): The cryo card relays value
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        relay = self.C.read_relays()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return relay

    def set_cryo_card_relay_bit(self, bitPosition, oneOrZero):
        """
        Sets a single cryo card relay to the value provided

        Args:
        -----
        bitPosition (int): Which bit to set.  Must be in [0-16].
        oneOrZero (int): What value to set the bit to.  Must be either 0 or 1.
        """
        assert (bitPosition in range(17)), 'bitPosition must be in [0,...,16]'
        assert (oneOrZero in [0,1]), 'oneOrZero must be either 0 or 1'
        currentRelay = self.get_cryo_card_relays()
        nextRelay = currentRelay & ~(1<<bitPosition)
        nextRelay = nextRelay | (oneOrZero<<bitPosition)
        self.set_cryo_card_relays(nextRelay)



    def set_cryo_card_relays(self, relay, write_log=False, enable_poll=False,
                             disable_poll=False):

        """
        Sets the cryo card relays

        Args:
        -----
        relays (hex): The cryo card relays
        """
        if write_log:
            self.log('Writing relay using cryo_card object. {}'.format(relay))

        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        self.C.write_relays(relay)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)


    def set_cryo_card_delatch_bit(self, bit, write_log=False, enable_poll=False,
                                  disable_poll=False):
        """
        Delatches the cryo card for a bit.

        Args:
        -----
        bit (int): The bit to temporarily delatch
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        if write_log:
            self.log('Setting delatch bit using cryo_card ' +
                'object. {}'.format(bit))
        self.C.delatch_bit(bit)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

    def set_cryo_card_hemt_ps_en(self, enable):
        """
        Set the cryo card HEMT power supply enable.

        Args:
        -----
        enable (bool): Power supply enable (True = enable, False = disable)
        Returns:
        --------
        None
        """
        if write_log:
            self.log('Writing HEMT PS enable using cryo_card object to {}'.format(enable))

        # Read the current enable word and merge this bit in position 0
        current_en_value = self.C.read_ps_en()
        if (enable):
            # Set bit 0
            new_en_value = current_en_value | 0x1
        else:
            # Clear bit 0
            new_value = current_en_value & 0x2

        # Write back the new value
        self.C.write_ps_en()

    def set_cryo_card_50k_ps_en(self, enable):
        """
        Set the cryo card 50k power supply enable.

        Args:
        -----
        enable (bool): Power supply enable (True = enable, False = disable)
        Returns:
        --------
        None
        """
        if write_log:
            self.log('Writing 50k PS enable using cryo_card object to {}'.format(enable))

        # Read the current enable word and merge this bit in position 1
        current_en_value = self.C.read_ps_en()
        if (enable):
            # Set bit 2
            new_en_value = current_en_value | 0x2
        else:
            # Clear bit 1
            new_value = current_en_value & 0x1

        # Write back the new value
        self.C.write_ps_en()

    def get_cryo_card_hemt_ps_en(self):
        """
        Set the cryo card HEMT power supply enable.

        Args:
        -----
        None
        Returns:
        --------
        enable (bool): Power supply enable (True = enable, False = disable)
        """

        # Read the power supply enable word and extract the status of bit 0
        en_value = self.C.read_ps_en()

        return (en_value & 0x1 == 0x1)

    def get_cryo_card_50k_ps_en(self):
        """
        Set the cryo card HEMT power supply enable.

        Args:
        -----
        None
        Returns:
        --------
        enable (bool): Power supply enable (True = enable, False = disable)
        """

        # Read the power supply enable word and extract the status of bit 1
        en_value = self.C.read_ps_en()

        return (en_value & 0x2 == 0x2)

    def get_cryo_card_ac_dc_mode(self):
        """
        Get the operation mode, AC or DC, based on the readback of the relays.

        Args:
        -----
        None

        Returns:
        --------
        mode(str): return string describing the operation mode. If the relays readback
        don't match, then the string 'ERROR' is returned.
        """

        # Read the relays status
        status = self.C.read_ac_dc_relay_status()

        # Both bit
        if status == 0x0:
            # When both readbacks are '0' we are in DC mode
            return("DC")
        elif status == 0x3:
            # When both readback are '1' we are in AC mode
            return("AC")
        else:
            # Anything else is an error
            return("ERROR")

    _smurf_to_gcp_stream = 'userConfig[0]'  # bit for streaming
    def get_user_config0(self, as_binary=False, **kwargs):
        """
        """
        val =  self._caget(self.timing_header +
                           self._smurf_to_gcp_stream, **kwargs)
        if as_binary:
            val = bin(val)

        return val

    def set_user_config0(self, val, as_binary=False, **kwargs):
        """
        """
        self._caput(self.timing_header +
                    self._smurf_to_gcp_stream, val, **kwargs)


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

    def set_smurf_to_gcp_writer(self, val, **kwargs):
        """
        Turns on or off data writer from smurf to GCP.
        This only accepts bools. Annoyingly the bit is
        0 for streaming and 1 for off. This function takes
        care of that, so True for streaming and False
        for off.
        """
        old_val = self.get_user_config0()
        if val == False:
            new_val = old_val | (2 << 1)
        elif val == True:
            new_val = old_val
            if old_val & 2 << 1 != 0:
                new_val = old_val & ~(2 << 1)
        self._caput(self.timing_header +
                    self._smurf_to_gcp_stream, new_val, **kwargs)

    def get_smurf_to_gcp_stream(self, **kwargs):
        """
        """
        return self._caget(self.timing_header +
                    self._smurf_to_gcp_stream, **kwargs)

    def set_smurf_to_gcp_writer(self, val, **kwargs):
        """
        Turns on or off data writer from smurf to GCP.
        This only accepts bools. Annoyingly the bit is
        0 for streaming and 1 for off. This function takes
        care of that, so True for streaming and False
        for off.
        """
        old_val = self.get_user_config0()
        if val == False:
            new_val = old_val | (2 << 1)
        elif val == True:
            new_val = old_val
            if old_val & 2 << 1 != 0:
                new_val = old_val & ~(2 << 1)
        self._caput(self.timing_header +
                    self._smurf_to_gcp_stream, new_val, **kwargs)

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

    def set_smurf_to_gcp_cfg_read(self, val, **kwargs):
        """
        If set to True, constantly reads smurf2mce.cfg at the MCE
        rate (~200 Hz). This is for updating IP address. Constantly
        reading the cfg file causes occasional dropped frames. So
        it should be set to False after the cfg is read.
        """
        read_bit = 3
        old_val = self.get_user_config0()
        if val:
            new_val = old_val | (1 << read_bit)
        elif ~val:
            new_val = old_val
            if old_val & 1 << read_bit != 0:
                new_val = old_val & ~(1 << read_bit)

        self._caput(self.timing_header +
                    self._smurf_to_gcp_stream, new_val, **kwargs)


    _num_rows = 'userConfig[2]'  # bit for num_rows
    def set_num_rows(self, val, **kwargs):
        """
        Sets num_rows in the SMuRF header. This is written in userConfig[2].

        Args:
        -----
        val (int): The value of num_rows
        """
        old = self._caget(self.timing_header +
                    self._num_rows)
        new = (old & 0xFFFF0000) + ((val & 0xFFFF))
        self._caput(self.timing_header +
                    self._num_rows, new, **kwargs)

    def set_num_rows_reported(self, val, **kwargs):
        """
        Sets num_rows_reported in the SMuRF header. This is written
        in userConfig[2].

        Args:
        -----
        val (int): The value of num_rows_reported
        """
        old = self._caget(self.timing_header +
                    self._num_rows)
        new = (old & 0x0000FFFF) + ((val & 0xFFFF) << 16)
        self._caput(self.timing_header +
                    self._num_rows, new, **kwargs)

    _row_len = 'userConfig[4]'
    def set_row_len(self, val, **kwargs):
        """
        Sets row_len in the SMuRF header. This is written in userConfig[4]

        Args:
        -----
        val (int): The value of row_len
        """
        old = self._caget(self.timing_header +
                    self._row_len)
        new = (old & 0xFFFF0000) + ((val & 0xFFFF))
        self._caput(self.timing_header +
                    self._row_len, new, **kwargs)

    def set_data_rate(self, val, **kwargs):
        """
        Sets data_rates in the SMuRF header. This is written in userConfig[4].

        Args:
        -----
        val (int): The value of data_rate
        """
        old = self._caget(self.timing_header +
                    self._row_len)
        new = (old & 0x0000FFFF) + ((val & 0xFFFF)<<16)
        self._caput(self.timing_header +
                    self._row_len, new, **kwargs)


    # Triggering commands
    _trigger_width = 'EvrV2TriggerReg[{}]:Width'
    def set_trigger_width(self, chan, val, **kwargs):
        """
        Mystery value that seems to make the timing system work
        """
        self._caput(self.trigger_root + self._trigger_width.format(chan),
                    val, **kwargs)


    _trigger_enable = 'EvrV2TriggerReg[{}]:Enable'
    def set_trigger_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root + self._trigger_enable.format(chan),
                   val, **kwargs)


    _trigger_channel_reg_enable = 'EvrV2ChannelReg[{}]:enable'
    def set_evr_channel_reg_enable(self, chan, val, **kwargs):
        """
        """
        self.log('set_evr_channel_reg_enable sets 2 bits. enable and Enable.')
        self._caput(self.trigger_root +
                    self._trigger_channel_reg_enable.replace('enable', 'Enable').format(chan), int(val),
                    **kwargs)
        self._caput(self.trigger_root +
                    self._trigger_channel_reg_enable.format(chan), val,
                    **kwargs)

    _trigger_reg_enable = 'EvrV2TriggerReg[{}]:enable'
    def set_evr_trigger_reg_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root +
                    self._trigger_reg_enable.format(chan), val,
                    **kwargs)

    _trigger_channel_reg_count = 'EvrV2ChannelReg[{}]:Count'
    def get_evr_channel_reg_count(self, chan, **kwargs):
        """
        """
        return self._caget(self.trigger_root +
                    self._trigger_channel_reg_count.format(chan),
                    **kwargs)

    _trigger_channel_reg_dest_sel = 'EvrV2ChannelReg[{}]:DestSel'
    def set_evr_trigger_channel_reg_dest_sel(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root +
                    self._trigger_channel_reg_dest_sel.format(chan),
                    val, **kwargs)


    _dac_reset = 'dacReset[{}]'
    def set_dac_reset(self, bay, dac, val, **kwargs):
        """
        Toggles the physical reset line to DAC. Set to 1 then 0

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        """
        self._caput(self.DBG.format(bay) + self._dac_reset.format(dac), val,
                    **kwargs)

    def get_dac_reset(self, bay, dac, **kwargs):
        """
        Reads the physical reset DAC register.  Will be either 0 or 1.

        Args:
        -----
        bay (int): Which bay [0 or 1].
        dac (int): Which DAC no. [0 or 1].
        """
        return self._caget(self.DBG.format(bay) + self._dac_reset.format(dac),
                           val, **kwargs)


    _debug_select = "DebugSelect[{}]"
    def set_debug_select(self, bay, val, **kwargs):
        """
        """
        self._caput(self.app_core + self._debug_select.format(bay),
                    val, **kwargs)
    
    def get_debug_select(self, bay, **kwargs):
        """
        """
        return self._caget(self.app_core + self._debug_select.format(bay),
                           **kwargs)

    _output_config = "OutputConfig[{}]"
    def set_crossbar_output_config(self, index, val, **kwargs):
        """
        """
        self._caput(self.crossbar + self._output_config.format(index),
                    val, **kwargs)
    
    def get_crossbar_output_config(self, index, **kwargs):
        """
        """
        return self._caget(self.crossbar + self._output_config.format(index),
                           **kwargs)    

    _timing_link_up = "RxLinkUp"
    def get_timing_link_up(self, **kwargs):
        """
        """
        return self._caget(self.timing_status +
                           self._timing_link_up, **kwargs)

    ## IN PROGRESS
    # assumes it's handed the decimal equivalent
    _lmk_reg = "LmkReg_0x{:04X}"
    def set_lmk_reg(self, bay, reg, val, **kwargs):
        """
        Can call like this get_lmk_reg(bay=0,reg=0x147,val=0xA)
        to see only hex as in gui.
        """
        self._caput(self.lmk.format(bay) + self._lmk_reg.format(reg),
                    val, **kwargs)    

    def get_lmk_reg(self, bay, reg, **kwargs):
        """
        Can call like this hex(get_lmk_reg(bay=0,reg=0x147))
        to see only hex as in gui.
        """
        return self._caget(self.lmk.format(bay) +
                           self._lmk_reg.format(reg), **kwargs)        
