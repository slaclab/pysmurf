#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    Readout for cryogenic detector systems. What is here today is the client
#    half: connect to a server, read and write it by semantic name, and run the
#    operations its tree offers. Building the register tree and serving it is the
#    other half, and it belongs here too -- it is not yet.
#
#        with cryodaq.connect('crate:4') as sess:
#            sess.set('band[0].tone.amplitude', 0, index=17)
#            sess.set('band[0].ops.gradient_descent.max_iters', 15)
#            proc = sess.call('band[0].ops.start_gradient_descent')
#            print(sess.node('band[0].ops.gradient_descent').Message.get())
#
#    Connecting needs rogue, which arrives with the smurf image rather than from
#    an index, so the session is imported on first use: `import cryodaq` and the
#    platform maps work anywhere, and the rogue import fails where it is really
#    needed instead of at the top of the package.
#
#    The client is a thin layer over rogue's own client: a session holds one, and
#    `session.root` is the server's tree, whole and unwrapped. What the client
#    adds is a name that means the same thing on every platform --
#    band[0].tone.amplitude, wherever that generation of firmware keeps it -- and
#    the small amount of bookkeeping a client needs: where it writes, where it
#    logs, what it publishes.
#
#    The mapping itself lives in cryodaq.platform, one module of data per
#    generation of hardware, and it is the only place a register path appears.
#    Applications -- which detectors are wired where, what a tuning run means for
#    them -- sit above and import this; nothing here imports an application.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import importlib
from typing import Any

from cryodaq import platform
from cryodaq._errors import ConnectError, CryodaqError, UnresolvedName
from cryodaq.platform import COMMAND, PROCESS, VALUE, PlatformMap

# Everything the session module offers, resolved on first use. The names are
# listed rather than discovered so that a typo is an AttributeError here and not
# a rogue import failure somewhere else.
_SESSION_NAMES = ('LOG_ERROR', 'LOG_INFO', 'LOG_USER', 'NullPublisher', 'Paths',
                  'Session', 'ValidationReport', 'connect', 'endpoint_of')


def __getattr__(name: str) -> Any:
    """Import the session on first use, so the package itself needs no rogue."""
    if name in _SESSION_NAMES:
        value = getattr(importlib.import_module('cryodaq._session'), name)
        globals()[name] = value             # once resolved, no lookup next time
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals()) + list(_SESSION_NAMES))


__all__ = [
    # the interface
    'connect', 'Session', 'Paths', 'ValidationReport', 'NullPublisher',
    'endpoint_of',
    # the map
    'platform', 'PlatformMap', 'VALUE', 'COMMAND', 'PROCESS',
    # errors
    'CryodaqError', 'ConnectError', 'UnresolvedName',
    # logging levels
    'LOG_ERROR', 'LOG_INFO', 'LOG_USER',
]
