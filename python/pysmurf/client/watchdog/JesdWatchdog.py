#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf watchdog module - JesdWatchdog class
#-----------------------------------------------------------------------------
# File       : pysmurf/watchdog/JesdWatchdog.py
# Created    : 2018-12-06
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import logging
import sys
import time
from datetime import datetime

import epics

class JesdWatchdog(object):
    def __init__(self, prefix):
        self.logfile = '/tmp/JesdWatchdog.log'
        logging.basicConfig(filename=self.logfile,level=logging.ERROR)

        self.prefix = prefix
        self.enabledPv     = epics.get_pv('SIOC:SMRF:ML00:AO001', callback=self.enableChanged, auto_monitor=True)
        self.enable        = self.enabledPv.get()
        self.jesdtxreset_thread = None
        self.jesdrxreset_thread = None
        self.counterPv     = epics.get_pv('SIOC:SMRF:ML00:AO001CNT')
        self.counterPvProc = epics.get_pv('SIOC:SMRF:ML00:AO001CNT.PROC')
        self.JesdRxValidPv = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdRx:DataValid', callback=self.jesdValidChanged, auto_monitor=True)
        self.JesdTxValidPv = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdTx:DataValid', callback=self.jesdValidChanged, auto_monitor=True)


    def enableChanged(self, pvname, value, *args, **kwargs):
        print("Enable changed to " + str(value))
        self.enable = value

    @staticmethod
    def jesdRXReset(prefix):
        logging.error('[%s] ' % str(datetime.now()) + ' JesdRx went down, will attempt to recover...')

        # for recovery
        PwrUpSysRef = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:LMK:PwrUpSysRef')
        JesdRxEnable = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdRx:Enable')

        #1. Toggle JesdRx:Enable 0x3F3 -> 0x0 -> 0x3F3
        JesdRxEnable.put(0x0)
        JesdRxEnable.put(0x3F3)
        # SYSREF is the last step
        PwrUpSysRef.put(1)

    @staticmethod
    def jesdTXReset(prefix):
        logging.error('[%s] ' % str(datetime.now()) + ' JesdTx went down, will attempt to recover...')

        # for recovery
        PwrUpSysRef = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:LMK:PwrUpSysRef')
        JesdTxEnable = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdTx:Enable')
        DAC0JesdRstN = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[0]:JesdRstN')
        DAC1JesdRstN = epics.get_pv(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[1]:JesdRstN')

        #1. Toggle JesdRx:Enable 0x3CF -> 0x0 -> 0x3CF
        JesdTxEnable.put(0x0)
        JesdTxEnable.put(0x3CF)

        #2. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[0]:JesdRstN 0x1 -> 0x0 -> 0x1
        DAC0JesdRstN.put(0x0)
        DAC0JesdRstN.put(0x1)

        #3. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[1]:JesdRstN 0x1 -> 0x0 -> 0x1
        DAC1JesdRstN.put(0x0)
        DAC1JesdRstN.put(0x1)

        # SYSREF is the last step
        PwrUpSysRef.put(1)

    def jesdValidChanged(self, pvname, value, *args, **kwargs):
        #print('[%s] ' % str(datetime.now()) + pvname + ' changed ; value=%s'%(str(value)))
        if self.enable == 1:
            if value == 0:
                # JesdRx
                if 'JesdRx' in pvname:
                    self.jesdrxreset_thread = epics.ca.CAThread(target=self.jesdRXReset, args=(self.prefix,))
                    self.jesdrxreset_thread.start()

                # JesdTx
                if 'JesdTx' in pvname:
                    self.jesdtxreset_thread = epics.ca.CAThread(target=self.jesdTXReset, args=(self.prefix,))
                    self.jesdtxreset_thread.start()

    def run(self):
        count  = self.counterPv.get()
        time.sleep(5)
        count1 = self.counterPv.get()
        if count != count1:
            logging.error('Another process incrementing the counter')
            return

        while True:
            time.sleep(1)
            self.counterPvProc.put(1)

        return





if __name__ == '__main__':
    prefix = 'test_epics'
    if len(sys.argv) > 1:
        prefix = sys.argv[1]

    wd = JesdWatchdog(prefix)
    wd.run()
