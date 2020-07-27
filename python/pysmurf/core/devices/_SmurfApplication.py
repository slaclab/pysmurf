#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Application
#-----------------------------------------------------------------------------
# File       : _SmurfApplication.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Application Device
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import os
import sys

import pyrogue

import pysmurf

class SmurfApplication(pyrogue.Device):
    """
    SMuRF Application Block
    """
    def __init__(self, **kwargs):

        pyrogue.Device.__init__(self, name="SmurfApplication", description='SMuRF Application Container', **kwargs)

        self.add(pyrogue.LocalVariable(
            name='SmurfVersion',
            description='PySMuRF Version',
            mode='RO',
            value=pysmurf.__version__))

        self.add(pyrogue.LocalVariable(
            name='SmurfDirectory',
            description='Path to the PySMuRF Python Files',
            value=os.path.dirname(pysmurf.__file__),
            mode='RO'))

        self.add(pyrogue.LocalVariable(
            name='StartupScript',
            description='PySMuRF Server Startup Script',
            value=sys.argv[0],
            mode='RO'))

        self.add(pyrogue.LocalVariable(
            name='StartupArguments',
            description='PySMuRF Server Startup Arguments',
            value=' '.join(sys.argv[1:]),
            mode='RO'))

        self.add(pyrogue.LocalVariable(
            name='SomePySmurfVariable',
            description='PySMuRF Variable Example',
            mode='RW',
            value=0, # Initial value determine variable type, (int, float, list, etc)
        ))

        # This variable will hold a list of the enabled bays. We set here the initial
        # value as a list of 2 elements, which will be its maximum size (as we have at
        # most 2 enabled bays). Its EPICS PV will be created with the initial size of
        # this list. This will generated an EPICS PV of size 2 in all cases, as currently
        # a bug in rogue does not allow to read list of size 1; also on the client side,
        # epics.caget will always return an numpy array even when only 1 bay is enabled.
        # We fill the list will invalid values; the list will be updated after the rogue
        # root is started (to wait for the PV to be created) with the correct values.
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

        # Command to call the 'AppTop.JesdHealth' command, using the wrapper method
        # 'root._check_jesd_health'. This method will verify if the command exist,
        # and it will update the 'JesdStatus' status register with the appropriate
        # value depending on the returned value of that command.
        self.add(pyrogue.LocalCommand(
            name='checkJesd',
            description='Check JESD health status',
            function=lambda arg: self.root._check_jesd_health()))

        # Status of the 'AppTop.JesdHealth' command. When the command is executed
        # this status is set to 'Checking' until the command finished, and then this
        # status is set to the returned value of the command (Locked/Unlocked)
        # accordingly.
        # By default this status is set to 'Unlocked', and the 'checkJesd' commands
        # need to be call once the system is started to updated it with the correct
        # value.
        # If the command does not exist, the status is set to "Not found".
        self.add(pyrogue.LocalVariable(
            name='JesdStatus',
            description='Jesd health status',
            mode='RO',
            enum={0:'Unlocked', 1:'Locked', 2:'Checking', 3:'Not found'},
            value=0))

