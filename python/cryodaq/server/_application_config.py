#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq ApplicationConfig Device
#-----------------------------------------------------------------------------
# File       : _application_config.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    The device a server carries its recorded configuration in: the resolved
#    values as JSON, their hash, and when they were written. A client that
#    connects later reads them back through `Session.resolved_config()` rather
#    than needing the file the first client was given.
#
#    Server-side code: a root attaches one (`root.add(ApplicationConfig())`) and the
#    operation that applies a configuration fills it through
#    `Session.record_config`.
#    The variables are excluded from the server's saved configuration and state
#    -- they describe those, and a snapshot that carried its own record would
#    be re-applied over the next one.
#
#    This is the one module in cryodaq that imports pyrogue at module level,
#    because it exists only where a server runs. The package does not import it;
#    a root does.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue

__all__ = ['ApplicationConfig', 'EXCLUDED_FROM']

# The variable groups a saved configuration and a saved state leave out.
EXCLUDED_FROM = ('NoConfig', 'NoState')


class ApplicationConfig(pyrogue.Device):
    """Where a server carries the configuration it was last given.

    Three string variables, all writable by a client: ``Resolved`` (the
    resolved values and their provenance as JSON), ``Hash`` (of the values) and
    ``WrittenAt`` (UTC). Empty until the configuring operation writes them.
    """

    def __init__(self, **kwargs):
        pyrogue.Device.__init__(self, name='ApplicationConfig',
                                description='The configuration this server was last given',
                                **kwargs)
        for name, description in (
                ('Resolved', 'Resolved configuration, as JSON'),
                ('Hash', 'SHA-256 of the resolved values'),
                ('WrittenAt', 'When the configuration was recorded, UTC')):
            variable = pyrogue.LocalVariable(name=name, description=description,
                                             mode='RW', value='')
            for group in EXCLUDED_FROM:
                variable.addToGroup(group)
            self.add(variable)
