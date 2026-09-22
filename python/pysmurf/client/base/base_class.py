#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf base module - SmurfBase class
#-----------------------------------------------------------------------------
# File       : pysmurf/base/base_class.py
# Created    : 2018-08-30
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import atexit

try:
    import pyrogue.interfaces
except ModuleNotFoundError:
    import warnings
    warnings.warn("Could not import pyrogue. Can only use offline mode.")

from cryodaq import platform
from pysmurf.client.command.cryo_card import CryoCard
from pysmurf.client.util.pub import Publisher
from .logger import SmurfLogger

class _DummyClient:
    """Dummy client to raise informative error messages
    when trying to access clients in offline mode."""
    def __init__(self, name=None):
        self.name = name

    def __getattr__(self, name):
        raise AttributeError(f"{self.name} not available.")


class SmurfBase:
    """
    Base class for common things

    Args
    ----
    log : log file or None, optional, default None
        The log file to write to. If None, creates a new log file.
    server_addr: str or None
        The server address
    server_port: int, optional, default 9000
        The server port on the server to connect to
    offline : bool, optional, default False
        Whether to run in offline mode (no rogue) or not. This
        will break many things. Default is False.
    pub_root : str or None, optional, default None
        Root of environment vars to set publisher options. If
        None, the default root will be `SMURFPUB_`.
    script_id : str or None, optional, default None
        Script id included with publisher messages. For example,
        the script or operation name.
    """

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

    def __init__(self, log=None, server_addr="localhost", server_port=9000, atca_port=9100,
                 atca_monitor=False, offline=False, pub_root=None, script_id=None, **kwargs):
        """
        """

        # Set up logging
        self.log = log
        if self.log is None:
            self.log = self.init_log(**kwargs)
        else:
            verb = kwargs.pop('verbose', None)
            if verb is not None:
                self.set_verbose(verb)

        self._server_addr = server_addr
        self._server_port = server_port
        self._atca_port = atca_port

        # suppress client logging errors
        # this doesn't work because rogue filters only can lower the level...
        #rogue.Logging.setFilter('pyrogue.ZmqClient', rogue.Logging.Critical)

        # connect to rogue servers
        if not offline:
            self._client = pyrogue.interfaces.VirtualClient(addr=self._server_addr, port=self._server_port)
            # Set a 30s timeout. And warn every 5s
            self._client.setTimeout(5000, 30000)  # ms
            # disable monitor thread that hangs on exit
            self._client._monEnable = False
            # ensure that client socket is closed
            atexit.register(self._client.stop)
            if atca_monitor:
                self._atca = pyrogue.interfaces.VirtualClient(addr=self._server_addr, port=self._atca_port)
                if self._atca.root is None:
                    self.log(f"Could not connect to ATCA monitor at port {self._atca_port}.")
                self._atca._monEnable = False
                atexit.register(self._atca.stop)
            else:
                self._atca = _DummyClient("ATCA monitor client")
        else:
            self._client = _DummyClient("OFFLINE: Server client")
            self._atca = _DummyClient("OFFLINE: ATCA monitor client")

        # If <pub_root>BACKEND environment variable is not set to 'udp', all
        # publish calls will be no-ops.
        self.pub = Publisher(env_root=pub_root, script_id=script_id)

        self.offline = offline
        if self.offline is True:
            self.log('Offline mode')

        # Register paths are resolved from semantic names through cryodaq.platform.
        # What is left here is the two prefixes another repository still builds paths
        # from, and the chain that reaches them:
        #
        #   cryo_root         sodetlib reads etaScanInProgress under it, via _cryo_root
        #   rtm_spi_max_root  sodetlib addresses the TES bias DACs under it
        #
        # Both go when those call sites move to public operations, and nothing else
        # depends on them.
        _app_core = 'AMCc.FpgaTopLevel.AppTop.AppCore.'

        self.sysgencryo = _app_core + 'SysgenCryo.'
        self.band_root = self.sysgencryo + 'Base[{}].'
        self.cryo_root = self.band_root + 'CryoChannels.'

        self.rtm_cryo_det_root = _app_core + 'RtmCryoDet.'
        self.rtm_spi_max_root = self.rtm_cryo_det_root + 'RtmSpiMax.'
        if offline:
            self.log('Offline mode, skipping CryoCard initialization')
            self.C = _DummyClient("OFFLINE: CryoCard client")
        else:
            # The cryostat card is reached over a serial link on the RTM, through a
            # pair of mailbox nodes. Where those are is a property of the platform,
            # so they are resolved through its map and handed over as nodes; the
            # card's own protocol is all that CryoCard then knows. It is given this
            # client's tree rather than opening a second connection to the same
            # endpoint, which is what it used to do.
            pmap = platform.identify(self._client.root)
            self.C = CryoCard(
                self._client.root.getNode(pmap.path('rtm.cryocard.read')),
                self._client.root.getNode(pmap.path('rtm.cryocard.write')),
                log=self.log,
            )

        self.freq_resp = {}

        # RTM slow DAC parameters (used, e.g., for TES biasing). The
        # DACs are AD5790 chips
        self._rtm_slow_dac_max_volt = 10. # Max unipolar DAC voltage,
                                        # in Volts
        self._rtm_slow_dac_nbits = 20
        # x2 because _rtm_slow_dac_max_volt is the maximum *unipolar*
        # voltage.  Units of Volt/bit
        self._rtm_slow_dac_bit_to_volt = (2*self._rtm_slow_dac_max_volt/
                                          (2**(self._rtm_slow_dac_nbits)))

        # LUT table length for arbitrary waveform generation
        self._lut_table_array_length = 2048

    def init_log(self, verbose=0, logger=SmurfLogger, logfile=None,
                 log_timestamp=True, log_prefix=None, **kwargs):
        """
        Initialize the logger from the input keyword arguments.

        Args
        ----
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

        Args
        ----
        band (int): The band to access

        Returns
        -------
        path (string) : The string to be passed to caget/caput to access
            the input band.
        '''
        return self.band_root.format(int(band))

    def _cryo_root(self, band):
        '''
        Helper function that returns the epics path to cryoroot.

        Args
        ----
        band (int): The band to access

        Returns
        -------
        path (string) : The string to be passed to caget/caput to access
            the input band.
        '''
        return self.cryo_root.format(int(band))
