#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : SMuRF UDP Receiver
#-----------------------------------------------------------------------------
# File       : _UdpReceiver.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF UDP receiver device.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import time
import threading
import rogue
import pyrogue

class KeepAlive(rogue.interfaces.stream.Master, threading.Thread):
    """
    Class used to keep alive the streaming data UDP connection.

    It is a Rogue Master device, which will be connected to the
    UDP Client slave.

    It will run a thread which will send an UDP packet every
    5 seconds to avoid the connection to be closed. After
    instantiate an object of this class, and connect it to the
    UDP Client slave, its 'start()' method must be called to
    start the thread itself.
    """
    def __init__(self):
        super().__init__()
        threading.Thread.__init__(self)

        # Define the thread as a daemon so it is killed as
        # soon as the main program exits.
        self.daemon = True

        # Request a 1-byte frame from the slave.
        self.frame = self._reqFrame(1, True)

        # Create a 1-byte element to be sent to the
        # slave. The content of the packet is not
        # important.
        self.ba = bytearray(1)

    def run(self):
        """
        This method is called the the class' 'start()'
        method is called.

        It implements an infinite loop that send an UDP
        packet every 5 seconds.
        """
        while True:
            self.frame.write(self.ba,0)
            self._sendFrame(self.frame)
            time.sleep(5)

class UdpReceiver(pyrogue.Device):
    """
    Class used to receive SMuRF streaming data over UDP,
    without RSSI, and with a keep alive mechanism.

    It is a Rogue master device, which will connected to the
    smurf-processor.
    """
    def __init__(self, ip_addr, port):
        super().__init__()
        # Create a Rogue UDP client.
        self._udp_receiver = rogue.protocols.udp.Client(ip_addr, port, True)

        # Create a KeepAlive object and connect it to the UDP client.
        self._keep_alive = KeepAlive()
        pyrogue.streamConnect(self._keep_alive, self._udp_receiver)

        # Start the KeepAlive thread
        self._keep_alive.start()

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._udp_receiver