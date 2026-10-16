#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Application Status Device
#-----------------------------------------------------------------------------
# File       : _application_status.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    The status block a readout server carries beside the FPGA: which bays are
#    enabled, whether the configuration procedure is running and whether it
#    succeeded, and -- on a platform with serial links to its converters -- the
#    JESD health as last checked. The device is named SmurfApplication, which is
#    the path clients have always read these at; an application adds its own
#    identity variables to the same device.
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

from cryodaq import platform

__all__ = ['ApplicationStatus']


class ApplicationStatus(pyrogue.Device):
    """Where a readout server reports its own state."""

    def __init__(self, pmap: platform.PlatformMap, application: str = '', **kwargs):
        pyrogue.Device.__init__(self, name=platform.STATUS_DEVICE_PATH.rsplit('.', 1)[1],
                                description='SMuRF Application Container', **kwargs)

        # Which application composed this server, so a client built for one
        # can refuse a server running another -- or none: the readout core
        # alone leaves it empty.
        self.add(pyrogue.LocalVariable(
            name=pmap.path('application.name').rsplit('.', 1)[1],
            description='The application this server was composed with; empty for the readout core alone',
            mode='RO',
            value=application))

        # A list of two so its size is fixed whether one or two bays are enabled,
        # filled in with the real list once the root has started: a published
        # variable's size is set when it is created, and the filler is 2 because
        # no bay has that index. (Published arrays of one element, or an empty
        # list, have not read back reliably through every client stack.)
        self.add(pyrogue.LocalVariable(
            name='EnabledBays',
            description='List of bays that are enabled',
            value=[2, 2],
            mode='RO'))

        self.add(pyrogue.LocalVariable(
            name='ConfiguringInProgress',
            description='The system configuration sequence is in progress',
            mode='RO',
            value=False))

        self.add(pyrogue.LocalVariable(
            name='SystemConfigured',
            description='The system was configured correctly',
            mode='RO',
            value=False))

        # Only a platform with serial links has a health to report. Set to
        # 'Checking' while the command runs, then to its verdict; 'Not found'
        # when the firmware release predates the check. Both are 'Unlocked'
        # until CheckJesd has run once after start -- the configuration
        # procedure runs it.
        if 'application.jesd_status' in pmap:
            self.add(pyrogue.LocalCommand(
                name=pmap.path('application.check_jesd').rsplit('.', 1)[1],
                description='Check JESD health status',
                function=lambda arg: self.root._check_jesd_health()))
            self.add(pyrogue.LocalVariable(
                name=pmap.path('application.jesd_status').rsplit('.', 1)[1],
                description='JESD health status (updates by calling the "CheckJesd" command)',
                mode='RO',
                enum={0: 'Unlocked', 1: 'Locked', 2: 'Checking', 3: 'Not found'},
                value=0))
