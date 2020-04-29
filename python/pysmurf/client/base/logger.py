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
import sys

__all__ = ['Logger', 'SmurfLogger']

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
