#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Publisher Interface
#-----------------------------------------------------------------------------
# File       : _SmurfPublisher.py
# Created    : 2019-10-31
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Publisher Interface Device
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

from pysmurf.client.util.pub import Publisher

class SmurfPublisher(object):
    """
    SMuRF Publisher Block
    """
    def __init__(self, root, pub_root=None, script_id=None):

        # If <pub_root>BACKEND environment variable is not set to 'udp', all
        # publish calls will be no-ops.
        self.pub = Publisher(env_root=pub_root, script_id=script_id)

        # Process root looking for variables part of the publish group
        for v in root.variableList:
            if v.inGroup('publish'):
                v.addVarListener(self._varListen)

    def _varListen(self, path, varVal):
        self.pub.publish(data=f'{path}={varVal.value}',msgType='general')

