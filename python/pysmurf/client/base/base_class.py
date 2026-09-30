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
import logging
import pathlib

import cryodaq
from pysmurf.client.command.cryo_card import CryoCard
from pysmurf.client.util.pub import Publisher
from .logger import SmurfLogger, SmurfLogHandler

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

        # If <pub_root>BACKEND environment variable is not set to 'udp', all
        # publish calls will be no-ops.
        self.pub = Publisher(env_root=pub_root, script_id=script_id)

        # connect to the rogue server
        if not offline:
            # The connection is a cryodaq session: one client, a 30 s request
            # timeout warning every 5 s, no link monitor (its thread hangs the
            # interpreter on exit), closed on the way out, and the platform
            # identified from the firmware the tree reports -- so a system no map
            # claims is refused here rather than at the first register a method
            # reaches. What the session logs arrives through the handler below,
            # since its level numbering runs the other way from SmurfLogger's.
            self._session = cryodaq.connect(
                f'{server_addr}:{server_port}', timeout=30.0, monitor=False,
                publisher=self.pub, logger=self._cryodaq_logger(),
                paths=self._session_paths())
            self.is_rfsoc = self._session.pmap.name == 'umux-rfsoc'
            if atca_monitor:
                # The shelf manager's monitor is a second rogue server with its
                # own tree, outside the platform map; it is reached directly.
                import pyrogue.interfaces
                self._atca = pyrogue.interfaces.VirtualClient(addr=self._server_addr, port=self._atca_port)
                if self._atca.root is None:
                    self.log(f"Could not connect to ATCA monitor at port {self._atca_port}.")
                self._atca._monEnable = False
                atexit.register(self._atca.stop)
            else:
                self._atca = _DummyClient("ATCA monitor client")
        else:
            # Offline there is no firmware to ask, and no register is reached.
            self._session = None
            self.is_rfsoc = False
            self._atca = _DummyClient("OFFLINE: ATCA monitor client")

        self.offline = offline
        if self.offline is True:
            self.log('Offline mode')

        if offline:
            self.log('Offline mode, skipping CryoCard initialization')
            self.C = _DummyClient("OFFLINE: CryoCard client")
        else:
            # The cryostat card is reached over a serial link on the RTM, through a
            # pair of mailbox nodes. Where those are is a property of the platform,
            # so they are resolved by name and handed over as nodes; the card's
            # own protocol is all that CryoCard then knows.
            self.C = CryoCard(
                self._session.node('rtm.cryocard.read'),
                self._session.node('rtm.cryocard.write'),
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

    @property
    def _client(self):
        """The rogue client the session holds.

        Kept for the methods that still ask the tree by register path; every
        other register is reached by name through the session.

        .. deprecated:: 11.5.0
            Goes with the methods scheduled for removal.
        """
        if self._session is None:
            return _DummyClient("OFFLINE: Server client")
        return self._session._client

    def _session_paths(self):
        """Where the session keeps its sidecar: the configuration's status directory.

        A client started without a configuration file has to look where the
        configuring client wrote, so with no file the packaged default's
        ``paths.status`` is used -- the same place a file that does not
        override it resolves to.
        """
        import dataclasses
        status = getattr(self, 'status_dir', None)
        if status is None:
            from pysmurf.client.config import DEFAULT
            status = cryodaq.config.load(DEFAULT).values['paths']['status']
        data = getattr(self, 'default_data_dir', None) or status
        return dataclasses.replace(cryodaq.Paths.under(data), status=pathlib.Path(status))

    def _cryodaq_logger(self):
        """A ``logging`` logger whose records land in this object's log.

        Private to this instance -- named after it -- and not propagated, so a
        second SmurfControl in the same process does not receive the first's
        session messages.
        """
        logger = logging.getLogger(f'cryodaq.pysmurf.{id(self)}')
        logger.handlers.clear()
        logger.addHandler(SmurfLogHandler(self.log, self._log_levels()))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger

    def _log_levels(self):
        """The named log levels, ``{'user': 0, 'error': 0, 'info': 1, 'task': 2}``."""
        levels = dict()
        for k in dir(self):
            if not k.startswith('LOG_'):
                continue
            levels[k.split('LOG_', 1)[1].lower()] = getattr(self, k)
        return levels

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
        log = logger(verbosity=verbose, logfile=logfile,
                     timestamp=timestamp, prefix=prefix,
                     levels=self._log_levels(), **kwargs)
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
