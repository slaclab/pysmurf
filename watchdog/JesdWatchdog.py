#!/usr/bin/env python

import sys
import os
import time
import epics
from datetime import datetime


class JesdWatchdog(object):
    def __init__(self, prefix):
        self.counterPv     = epics.PV('SIOC:SMRF:ML00:AO001CNT')
        self.counterPvProc = epics.PV('SIOC:SMRF:ML00:AO001CNT.PROC')
        self.enabledPv     = epics.PV('SIOC:SMRF:ML00:AO001', callback=self.enableChanged, auto_monitor=True)
        self.JesdRxValidPv = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdRx:DataValid', callback=self.jesdValidChanged, auto_monitor=True)
        self.JesdTxValidPv = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdTx:DataValid', callback=self.jesdValidChanged, auto_monitor=True)

        # for recovery
        self.PwrUpSysRef = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:LMK:PwrUpSysRef')
        self.JesdRxEnable = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdRx:Enable')
        self.JesdTxEnable = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppTopJesd[0]:JesdTx:Enable')
        self.DAC0JesdRstN = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[0]:JesdRstN')
        self.DAC1JesdRstN = epics.PV(prefix + ':AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[1]:JesdRstN')

        self.enable        = self.enabledPv.get()

    def enableChanged(self, pvname, value, *args, **kwargs):
        self.enable = value

    def jesdValidChanged(self, pvname, value, *args, **kwargs):
        #print('[%s] ' % str(datetime.now()) + pvname + ' changed ; value=%s'%(str(value)))
        if self.enable == 1:
            if value == 0:
                
                # JesdRx
                if 'JesdRx' in pvname:
                    print('[%s] ' % str(datetime.now()) + ' JesdRx went down, will attempt to recover...')
                    #1. Toggle JesdRx:Enable 0x3F3 -> 0x0 -> 0x3F3
                    self.JesdRxEnable.put(0x0)
                    self.JesdRxEnable.put(0x3F3)

                # JesdTx
                if 'JesdTx' in pvname:
                    print('[%s] ' % str(datetime.now()) + ' JesdTx went down, will attempt to recover...')
                    #1. Toggle JesdRx:Enable 0x3CF -> 0x0 -> 0x3CF
                    self.JesdTxEnable.put(0x0)
                    self.JesdTxEnable.put(0x3CF)
                    
                    #2. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[0]:JesdRstN 0x1 -> 0x0 -> 0x1
                    self.DAC0JesdRstN.put(0x0)
                    self.DAC0JesdRstN.put(0x1)
                    
                    #3. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:DAC[1]:JesdRstN 0x1 -> 0x0 -> 0x1
                    self.DAC1JesdRstN.put(0x0)
                    self.DAC1JesdRstN.put(0x1)
                
                # SYSREF is the last step for both
                self.PwrUpSysRef.put(1)


    def run(self):
        count  = self.counterPv.get()
        time.sleep(5)
        count1 = self.counterPv.get()
        if count != count1:
            print('Another process incrementing the counter')
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
