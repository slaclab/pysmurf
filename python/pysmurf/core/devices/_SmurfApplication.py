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

import pyrogue
import pysmurf
import os

class SmurfApplication(pyrogue.Device):
    """
    SMuRF Application Block
    """
    def __init__(self, **kwargs):
        pyrogue.Device.__init__(self, name="SmurfApplication", description='SMuRF Application Container', **kwargs)

        self.add(pyrogue.LocalVariable(
            name='SmurfVersion',
            description='SMURF Version Field',
            mode='RO',
            value="Unknown"))

        self.add(pyrogue.LocalVariable(
            name='SmurfDirectory',
            description='Path to SMURF Python Files',
            value=os.path.dirname(pysmurf.__file__),
            mode='RO'))

        self.add(pyrogue.LocalVariable(
            name='SomePySmurfVariable',
            description='PySmurf Variable Example',
            mode='RW',
            value=0, # Initial value determine variable type, (int, float, list, etc)
        ))

