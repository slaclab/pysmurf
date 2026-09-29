#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf base module - Logger class
#-----------------------------------------------------------------------------
# File       : pysmurf/base/logger.py
# Created    : 2018-08-29
#-----------------------------------------------------------------------------
# This is adapted from spider_tools.  Originally written by
# M. Hasselfield and ported by S. Rahlin
#-----------------------------------------------------------------------------
import datetime as dt
import logging
import sys

__all__ = ['Logger', 'SmurfLogger', 'SmurfLogHandler']

class Logger(object):
    """Basic prioritized logger, by M. Hasselfield."""

    def __init__(self, verbosity=0, indent=True, logfile=None):
        self.v = verbosity
        self.indent = indent
        self.set_logfile(logfile)

    def set_verbosity(self, level):
        """
        Change the verbosity level of the logger.
        """
        self.v = level

    set_verbose = set_verbosity

    def set_logfile(self, logfile=None):
        """
        Change the location where logs are written.  If logfile is None,
        log to STDOUT.
        """
        if hasattr(self, 'logfile') and self.logfile != sys.stdout:
            self.logfile.close()
        if logfile is None:
            self.logfile = sys.stdout
        else:
            self.logfile = open(logfile, 'a', 1)

    def format(self, s, level=0):
        """
        Format the input for writing to the logfile.
        """
        s = str(s)
        if self.indent:
            s = ' ' * level + s
        s += '\n'
        return s

    def write(self, s, level=0):
        if level <= self.v:
            self.logfile.write(self.format(s, level))

    def __call__(self, *args, **kwargs):
        """
        Log a message.

        Args
        ----
        msg : string
            The message to log.
        level : int, optional
            The verbosity level of the message.  If at or below the set level,
            the message will be logged.
        """
        return self.write(*args, **kwargs)

class SmurfLogger(Logger):
    """
    Basic logger with timestamps and named logging levels.
    """

    def __init__(self, **kwargs):
        self.timestamp = kwargs.pop('timestamp', True)
        self.prefix = kwargs.pop('prefix', None)
        self.levels = kwargs.pop('levels', {})
        kwargs.update(verbosity=self.get_level(kwargs.get('verbosity')))
        super(SmurfLogger, self).__init__(**kwargs)

    def set_verbosity(self, v):
        super(SmurfLogger, self).set_verbosity(self.get_level(v))

    def get_level(self, v):
        if v is None:
            return 0
        v = self.levels.get(v, v)
        if not isinstance(v, int):
            raise ValueError(f'Unrecognized logging level {v}')
        return v

    def format(self, s, level=0):
        """
        Format the input for writing to the logfile.
        """
        if self.prefix:
            s = f'{self.prefix}{s}'
        else:
            s = f'{s}'
        if self.timestamp:
            stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S%Z')
            s = f'[ {stamp} ]  {s}'
        return super(SmurfLogger, self).format(s, self.get_level(level))

    # root argument added as hack so that MPI/non-MPI code can get along
    def write(self, s, level=0, root=False):
        super(SmurfLogger, self).write(s, self.get_level(level))


class SmurfLogHandler(logging.Handler):
    """Deliver records from a standard ``logging`` logger to a SmurfLogger.

    The two loggers number their levels in opposite directions. A SmurfLogger
    level is a verbosity threshold -- a message shows when its level is at or
    below the configured verbosity, so ``0`` always shows and ``2`` is the
    quietest. ``logging`` runs the other way: ``ERROR`` (40) is louder than
    ``INFO`` (20). Handing a SmurfLogger straight to a library that logs
    through ``logging`` would send every error through the wrong comparison
    and the log would go dark, errors first. This handler is the translation,
    in the one direction ever needed.

    Parameters
    ----------
    smurf_log : SmurfLogger
        Where the records go.
    levels : dict
        The SmurfLogger's named levels (``user``, ``error``, ``info``,
        ``task``), as ``SmurfBase.init_log`` collects them.
    """

    def __init__(self, smurf_log, levels):
        super().__init__()
        self._log = smurf_log
        # Highest logging level first; a record maps to the first row it reaches.
        self._table = (
            (logging.ERROR, levels['error']),
            (25, levels['user']),           # cryodaq's USER level
            (logging.INFO, levels['info']),
            (logging.DEBUG, levels['task']),
        )

    def translate(self, levelno):
        """The SmurfLogger level a ``logging`` level number maps to."""
        for threshold, level in self._table:
            if levelno >= threshold:
                return level
        return self._table[-1][1]

    def emit(self, record):
        try:
            self._log(record.getMessage(), self.translate(record.levelno))
        except Exception:
            self.handleError(record)
