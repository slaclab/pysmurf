from epics import PV
import os
import sys

slots=[2,3]
cadence_sec=5

pvs = {}
for slot in slots:
    pvs[slot]={}
    pvs[slot]['sofCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:sofCount')
    pvs[slot]['eofCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:eofCount')
    pvs[slot]['fidCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:FidCount')
    pvs[slot]['rxClkCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxClkCount')
    pvs[slot]['rxRstCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxRstCount')
    pvs[slot]['crcErrCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:CrcErrCount')
    pvs[slot]['rxDecErrCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxDecErrCount')
    pvs[slot]['rxDspErrCount'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxDspErrCount')
    pvs[slot]['rxLinkUp'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:TimingFrameRx:RxLinkUp')
    
def get_timing_packet_stats(pvs,timeout=5):
    result = {}
    for k in pvs.keys():
        result[k]=pvs[k].get()
    return result

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

            try:
                result=get_timing_packet_stats(pvs[slot])
                entry=fmt.format([f'smurf_server_s{slot}',
                                  ctime,
                                  result['crcErrCount'],
                                  result['rxDecErrCount'],
                                  result['rxDspErrCount'],
                                  result['sofCount'],
                                  result['eofCount'],
                                  result['fidCount'],
                                  result['rxClkCount'],
                                  result['rxRstCount'],
                                  result['rxLinkUp']])
                print(entry)
                logf.write(entry)                
            except:
                print("Cagets stalled!")
                
        time.sleep(cadence_sec)
