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
import re
import sys
import time
from datetime import datetime

import epics

# Matches the bay index in PV names like
# "...:AppTopJesd[1]:JesdRx:DataValid".
_BAY_RE = re.compile(r'AppTopJesd\[(\d+)\]')


class JesdWatchdog(object):
    def __init__(self, prefix, bays):
        self.logfile = '/tmp/JesdWatchdog.log'
        logging.basicConfig(filename=self.logfile, level=logging.ERROR)

        self.prefix = prefix
        self.bays = list(bays)
        self.enabledPv     = epics.get_pv('SIOC:SMRF:ML00:AO001', callback=self.enableChanged, auto_monitor=True)
        self.enable        = self.enabledPv.get()
        self.jesdtxreset_thread = {bay: None for bay in self.bays}
        self.jesdrxreset_thread = {bay: None for bay in self.bays}
        self.counterPv     = epics.get_pv('SIOC:SMRF:ML00:AO001CNT')
        self.counterPvProc = epics.get_pv('SIOC:SMRF:ML00:AO001CNT.PROC')

        # Keep PV references in dicts so callbacks stay registered for the
        # lifetime of the watchdog. One Rx/Tx pair per enabled bay.
        self.JesdRxValidPv = {}
        self.JesdTxValidPv = {}
        for bay in self.bays:
            rx_pv = f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdRx:DataValid'
            tx_pv = f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdTx:DataValid'
            self.JesdRxValidPv[bay] = epics.get_pv(rx_pv, callback=self.jesdValidChanged, auto_monitor=True)
            self.JesdTxValidPv[bay] = epics.get_pv(tx_pv, callback=self.jesdValidChanged, auto_monitor=True)

    def enableChanged(self, pvname, value, *args, **kwargs):
        print("Enable changed to " + str(value))
        self.enable = value

    @staticmethod
    def jesdRXReset(prefix, bay):
        logging.error(f'[{datetime.now()}] '
                      f' JesdRx went down on bay={bay}, will attempt to recover...')

        # for recovery
        PwrUpSysRef = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[{bay}]:LMK:PwrUpSysRef')
        JesdRxEnable = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdRx:Enable')

        #1. Toggle JesdRx:Enable 0x3F3 -> 0x0 -> 0x3F3
        JesdRxEnable.put(0x0)
        JesdRxEnable.put(0x3F3)
        # SYSREF is the last step
        PwrUpSysRef.put(1)

    @staticmethod
    def jesdTXReset(prefix, bay):
        logging.error(f'[{datetime.now()}] '
                      f' JesdTx went down on bay={bay}, will attempt to recover...')

        # for recovery
        PwrUpSysRef = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[{bay}]:LMK:PwrUpSysRef')
        JesdTxEnable = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdTx:Enable')
        DAC0JesdRstN = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[{bay}]:DAC[0]:JesdRstN')
        DAC1JesdRstN = epics.get_pv(f'{prefix}:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[{bay}]:DAC[1]:JesdRstN')

        #1. Toggle JesdRx:Enable 0x3CF -> 0x0 -> 0x3CF
        JesdTxEnable.put(0x0)
        JesdTxEnable.put(0x3CF)

        #2. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[bay]:DAC[0]:JesdRstN 0x1 -> 0x0 -> 0x1
        DAC0JesdRstN.put(0x0)
        DAC0JesdRstN.put(0x1)

        #3. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[bay]:DAC[1]:JesdRstN 0x1 -> 0x0 -> 0x1
        DAC1JesdRstN.put(0x0)
        DAC1JesdRstN.put(0x1)

        # SYSREF is the last step
        PwrUpSysRef.put(1)

    def jesdValidChanged(self, pvname, value, *args, **kwargs):
        if self.enable != 1 or value != 0:
            return

        match = _BAY_RE.search(pvname)
        if match is None:
            logging.error(f'[{datetime.now()}] could not parse bay from pvname={pvname}')
            return
        bay = int(match.group(1))

        # JesdRx
        if 'JesdRx' in pvname:
            self.jesdrxreset_thread[bay] = epics.ca.CAThread(
                target=self.jesdRXReset, args=(self.prefix, bay))
            self.jesdrxreset_thread[bay].start()

        # JesdTx
        if 'JesdTx' in pvname:
            self.jesdtxreset_thread[bay] = epics.ca.CAThread(
                target=self.jesdTXReset, args=(self.prefix, bay))
            self.jesdtxreset_thread[bay].start()

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

    if len(sys.argv) > 2:
        bays = [int(b) for b in sys.argv[2:]]
    else:
        bays = [0]

    wd = JesdWatchdog(prefix, bays)
    wd.run()
