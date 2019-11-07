#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Group Setup Utility
#-----------------------------------------------------------------------------
# File       : _SmurfPublisher.py
# Created    : 2019-10-31
#-----------------------------------------------------------------------------
# Description:
#    Setup groups for variables & devices
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

VariableGroups = {'root.LocalTime':                            ['publish','stream'],
                  'root.SmurfApplication.SomePySmurfVariable': ['publish','stream']}


def setupGroups(root):
    for k,v in VariableGroups.items():

        # Get node
        n = root.getNode(k)

        # Add to groups
        for g in v:
            n.addToGroup(g)


