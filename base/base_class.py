import numpy as np
from .logger import SmurfLogger
from pysmurf.command.cryo_card import CryoCard

class SmurfBase(object):
    '''
    Base class for common things
    '''

    _base_args = ['verbose', 'logfile', 'log_timestamp', 'log_prefix',
                  'load_configs', 'log', 'layout']

    LOG_USER = 0
    """
    Default log level for user code. DO NOT USE in library
    """
    LOG_ERROR = 0   # deliberately same as LOG_USER
    """
    Only log errors
    """
    LOG_INFO = 1
    """
    Extra high-level information. Configuration notices that happen once
    """
    LOG_TASK = 2
    """
    Overall progress on a task
    """

    def __init__(self, log=None, epics_root=None, offline=False, **kwargs):
        # Set up logging
        self.log = log
        if self.log is None:
            self.log = self.init_log(**kwargs)
        else:
            verb = kwargs.pop('verbose', None)
            if verb is not None:
                self.set_verbose(verb)

        self.offline = offline
        if self.offline == True:
            self.log('Offline mode')


        # Setting paths for easier commands - Is there a better way to do this
        # than just hardcoding paths? This needs to be cleaned up somehow
        self.epics_root = epics_root
        self.amc_root = epics_root + ':AMCc:'
        self.app_core = self.epics_root + ':AMCc:FpgaTopLevel:AppTop:AppCore:'
        self.microwave_mux_core = self.app_core + 'MicrowaveMuxCore[0]:'
        self.DBG = self.microwave_mux_core + 'DBG:'
        self.sysgencryo = self.app_core + 'SysgenCryo:'
        self.band_root = self.sysgencryo + 'Base[{}]:'
        self.adc_root = self.sysgencryo + 'CryoAdcMux:'
        self.cryo_root = self.band_root + 'CryoChannels:'
        self.channel_root = self.cryo_root + 'CryoChannel[{}]:'
        self.dac_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[{}]:'
        self.att_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:'
        self.timing_header = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppCore:TimingHeader:'
        self.streaming_root = self.epics_root + ':AMCc:streamingInterface:'
        self.sysref = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[{}]:'

        # Tx -> DAC , Rx <- ADC
        self.axi_version = self.epics_root + \
            ':AMCc:FpgaTopLevel:AmcCarrierCore:AxiVersion:'
        self.waveform_engine_buffers_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:'+ \
            'AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[0]:' + \
            'WaveformEngineBuffers:'
        self.stream_data_writer_root = self.epics_root + \
            ':AMCc:streamDataWriter:'
        self.jesd_tx_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdTx:'
        self.jesd_rx_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdRx:'
        self.daq_mux_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:'

        # RTM paths
        self.rtm_cryo_det_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AppTop:AppCore:RtmCryoDet:'
        self.rtm_spi_root = self.rtm_cryo_det_root + \
            'RtmSpiSr:'
        self.rtm_spi_max_root = self.rtm_cryo_det_root + \
            'RtmSpiMax:'
        self.rtm_spi_cryo_root = self.rtm_cryo_det_root + \
            'SpiCryo:'

        # Timing paths
        self.trigger_root = self.epics_root + \
            ':AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:' + \
            'EvrV2CoreTriggers:'


        self.C = CryoCard(self.rtm_spi_cryo_root + 'read', 
            self.rtm_spi_cryo_root + 'write')
        self.freq_resp = {}

    def init_log(self, verbose=0, logger=SmurfLogger, logfile=None,
                 log_timestamp=True, log_prefix=None, **kwargs):
        """
        Initialize the logger from the input keyword arguments.

        Arguments
        ---------
        logger : logging class, optional
            Class to initialize, should be a subclass of SmurfLogger
            or equivalent.
        verbose : bool, int, or string; optional
            Verbosity level, non-negative.  Default: 0 (print user-level
            messages only). String options are 'info', 'time', 'gd', or 'samp'.
        logfile : string, optional
            Logging output filename.  Default: None (print to sys.stdout)
        log_timestamp : bool, optional
            If True, add timestamps to log entries. Default: True
        log_prefix : string, optional
            If supplied, this prefix will be pre-pended to log strings,
            before the timestamp.

        Returns
        -------
        log : log object
            Initialized logging object
        """
        if verbose is None:
            verbose = 0

        timestamp = log_timestamp
        prefix = log_prefix
        levels = dict()
        for k in dir(self):
            if not k.startswith('LOG_'):
                continue
            v = getattr(self, k)
            name = k.split('LOG_', 1)[1].lower()
            levels[name] = v
        log = logger(verbosity=verbose, logfile=logfile,
                     timestamp=timestamp, prefix=prefix,
                     levels=levels, **kwargs)
        return log

    def set_verbose(self, level):
        """
        Change verbosity level.  Can be an integer or a string name.
        Valid strings are 'info', 'time', 'gd' or 'samp'.
        """
        self.log.set_verbosity(level)

    def set_logfile(self, logfile=None):
        """
        Change the location where logs are written.  If logfile is None,
        log to STDOUT.
        """
        self.log.set_logfile(logfile)

    def _band_root(self, band):
        '''
        Helper function that returns the epics path to a band.

        Args:
        -----
        band (int): The band to access

        Returns:
        --------
        path (string) : The string to be passed to caget/caput to access
            the input band.
        '''
        return self.band_root.format(int(band))

    def _cryo_root(self, band):
        '''
        Helper function that returns the epics path to cryoroot.

        Args:
        -----
        band (int): The band to access

        Returns:
        --------
        path (string) : The string to be passed to caget/caput to access
            the input band.
        '''
        return self.cryo_root.format(int(band))

    def _channel_root(self, band, channel):
        """
        Helper function that returns the epics path to channel root.

        Args:
        -----
        band (int) : The band to access
        channel (int) : The channel to access.

        Returns:
        --------
        path (string) : The string to be passed to caget/caput to access
            the input band.
        """
        return self.channel_root.format(int(band), int(channel))
