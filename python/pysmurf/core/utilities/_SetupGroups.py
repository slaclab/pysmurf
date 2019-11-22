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

# Below is a dictionary with the key being the path to a Rogue Device or Variable
# The 'groups' entry provides a list of groups to add the Device/Variable to. If the
# path points to a device, the group will be added recursively to all devices and
# variables deeper in the path.
# The 'pollInterval' entry provides an optional value which will be used to update
# the polling interface if the path points to a variable. The poll interval value
# is in seconds. Use None to leave interval unchanged, 0 to disable polling.

VariableGroups = {
    'root.RogueVersion'     : {'groups' : ['publish','stream'], 'pollInterval': None},
    'root.RogueDirectory'   : {'groups' : ['publish','stream'], 'pollInterval': None},
    'root.SmurfApplication' : {'groups' : ['publish','stream'], 'pollInterval': None},
    'root.SmurfProcessor'   : {'groups' : ['publish','stream'], 'pollInterval': None},
    }


def setupGroups(root):
    for k,v in VariableGroups.items():

        # Get node
        n = root.getNode(k)

        # Did we find the node?
        if n is not None:

            # Add to each group
            for grp in v['groups']:
                n.addToGroup(grp)

            # Update poll interval if provided.
            if v['pollInterval'] is not None and n.isinstance(pyrogue.BaseVariable):
                n.pollInterval = v['pollInterval']

        else:
            print(f"setupGroups: Warning: {k} not found!")

