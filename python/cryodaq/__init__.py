#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    A client for cryogenic detector readout systems: connect to a server, read
#    and write it by semantic name, and run the operations its tree offers.
#
#        with cryodaq.connect('crate:4') as sess:
#            sess.set('band[0].tone.amplitude', 0, index=17)
#            sess.set('band[0].ops.gradient_descent.max_iters', 15)
#            proc = sess.call('band[0].ops.start_gradient_descent')
#            print(sess.node('band[0].ops.gradient_descent').Message.get())
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

from cryodaq import platform
from cryodaq._errors import ConnectError, CryodaqError, UnresolvedName
from cryodaq._session import (LOG_ERROR, LOG_INFO, LOG_USER, NullPublisher,
                              Paths, Session, ValidationReport, connect,
                              endpoint_of)
from cryodaq.platform import COMMAND, PROCESS, VALUE, PlatformMap

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
