import numpy as np
import os
import sys
from logger import SmurfLogger
# import smurf_control.command
from smurf.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin

class SmurfControl(SmurfCommandMixin):
    '''
    Base class for controlling Smurf
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

    def __init__(self, log=None, **kwargs):
        # Set up logging
        self.log = log
        if self.log is None:
            self.log = self.init_log(**kwargs)
        else:
            verb = kwargs.pop('verbose', Nonse)
            if verb is not None:
                self.set_verbose(verb)

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