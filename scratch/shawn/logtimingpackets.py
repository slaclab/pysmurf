from epics import PV
import os
import sys

slots=[2,3]
cadence_sec=5

def get_timing_packet_stats(slot,timeout=5):
    sofCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:sofCount')
    eofCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:eofCount')
    fidCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:FidCount')
    rxClkCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxClkCount')
    rxRstCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxRstCount')            
    crcErrCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:CrcErrCount')
    rxDecErrCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxDecErrCount')
    rxDspErrCountPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxDspErrCount')
    rxLinkUpPV=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxLinkUp')    
    return (sofCountPV.get(timeout=timeout,as_string=False),
            eofCountPV.get(timeout=timeout,as_string=False),
            fidCountPV.get(timeout=timeout,as_string=False),
            rxClkCountPV.get(timeout=timeout,as_string=False),
            rxRstCountPV.get(timeout=timeout,as_string=False),
            crcErrCountPV.get(timeout=timeout,as_string=False),
            rxDecErrCountPV.get(timeout=timeout,as_string=False),
            rxDspErrCountPV.get(timeout=timeout,as_string=False),
            rxLinkUpPV.get(timeout=timeout,as_string=False),)

import time
ctime0=int(time.time())

fmt='{0[0]:<20}{0[1]:<20}{0[2]:<20}{0[3]:<20}{0[4]:<20}{0[5]:<20}{0[6]:<20}{0[7]:<20}{0[8]:<20}{0[9]:<20}{0[10]:<4}\n'

hdr=fmt.format(['epics_root',
                'ctime',
                'CrcErrCount',
                'RxDecErrCount',
                'RxDspErrCount',
                'sofCount',
                'eofCount',
                'FidCount',
                'RxClkCount',
                'RxRstCount',
                'RxLinkUp'])

filepath='/data/smurf_data/'
filename=os.path.join(filepath,f'{ctime0}_timingpackets.log')

print(f'-> Logging to {filename}')

if not os.path.exists(filename):
    with open(filename,'a') as logf:
        logf.write(hdr)

while True:
    with open(filename,'a') as logf:
        for slot in slots:
            ctime=int(time.time())

            (sofCount,eofCount,fidCount,rxClkCount,rxRstCount,crcErrCount,rxDecErrCount,rxDspErrCount,rxLinkUp)=(None,None,None,None,None,None,None,None,None)
            try:
                (sofCount,eofCount,fidCount,rxClkCount,rxRstCount,crcErrCount,rxDecErrCount,rxDspErrCount,rxLinkUp)=get_timing_packet_stats(slot)
            except:
                print("Cagets stalled!")
                
            entry=fmt.format([f'smurf_server_s{slot}',
                              ctime,
                              crcErrCount,
                              rxDecErrCount,
                              rxDspErrCount,
                              sofCount,
                              eofCount,
                              fidCount,
                              rxClkCount,
                              rxRstCount,
                              rxLinkUp])
            print(entry)
            logf.write(entry)
        time.sleep(cadence_sec)
