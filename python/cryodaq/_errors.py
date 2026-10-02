#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Error Model
#-----------------------------------------------------------------------------
# File       : _errors.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    The exceptions cryodaq raises. A failed lookup raises; nothing returns None
#    or a sentinel to mean "could not". There are four, one per thing cryodaq
#    itself can be wrong about: the connection, the name, and a configuration
#    that cannot be resolved from its layers -- and nothing else. A register
#    that refuses a write, or a process that fails, reports through rogue and
#    is not re-wrapped here.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from typing import Optional

__all__ = ['CryodaqError', 'ConnectError', 'UnresolvedName', 'ConfigError']


class CryodaqError(Exception):
    """Base class of every error cryodaq raises."""


class ConnectError(CryodaqError):
    """The endpoint could not be parsed, no server answers it, or its tree is unknown."""


class UnresolvedName(CryodaqError):
    """A semantic name is not in the platform map, or its register is not in this tree.

    Parameters
    ----------
    name : str
        The name as given.
    pattern : str, optional
        The name pattern or register path that was tried, when there was one.
    reason : str
        Why it did not resolve, in a few words.
    """

    def __init__(self, name: str, pattern: Optional[str] = None, reason: str = ''):
        self.name = name
        self.pattern = pattern
        self.reason = reason
        tried = f" (tried {pattern!r})" if pattern else ''
        why = f": {reason}" if reason else ''
        super().__init__(f"cannot resolve {name!r}{tried}{why}")


class ConfigError(CryodaqError):
    """A configuration could not be resolved from its layers.

    Raised for what is wrong with the layering itself -- a file that cannot be
    read or parsed, an inheritance chain that loops, a layer that is not a
    mapping. What the values *mean* is the application's to judge, and its
    validator raises its own errors.

    Parameters
    ----------
    file : str
        The layer the fault is in.
    key : str, optional
        The key, dotted, when the fault is at one.
    reason : str
        What is wrong, in a few words.
    """

    def __init__(self, file: str, key: Optional[str] = None, reason: str = ''):
        self.file = file
        self.key = key
        self.reason = reason
        where = f"{file}:{key}" if key else file
        why = f": {reason}" if reason else ''
        super().__init__(f"cannot resolve configuration at {where}{why}")
