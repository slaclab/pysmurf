from epics import PV
import os
import sys

slots=[2,3,4,5,6,7]
cadence_sec=60

def get_packet_stats(slot,timeout=5):
    frameCntPV=PV(f'smurf_server_s{slot}:AMCc:SmurfProcessor:FrameRxStats:FrameCnt')
    frameSizePV=PV(f'smurf_server_s{slot}:AMCc:SmurfProcessor:FrameRxStats:FrameSize')
    frameLossCntPV=PV(f'smurf_server_s{slot}:AMCc:SmurfProcessor:FrameRxStats:FrameLossCnt')
    frameOutOrderCntPV=PV(f'smurf_server_s{slot}:AMCc:SmurfProcessor:FrameRxStats:FrameOutOrderCnt')
    badFrameCntPV=PV(f'smurf_server_s{slot}:AMCc:SmurfProcessor:FrameRxStats:BadFrameCnt')        
    return (frameCntPV.get(timeout=timeout),
            frameSizePV.get(timeout=timeout),
            frameLossCntPV.get(timeout=timeout),
            frameOutOrderCntPV.get(timeout=timeout),
            badFrameCntPV.get(timeout=timeout),)

import time
ctime0=int(time.time())

fmt='{0[0]:<20}{0[1]:<20}{0[2]:<20}{0[3]:<20}{0[4]:<20}{0[5]:<20}{0[6]:<20}\n'

hdr=fmt.format(['epics_root','ctime','FrameCnt','FrameSize','FrameLossCnt','FrameOutOrderCnt','BadFrameCnt'])

filepath='/data/smurf_data/simonsobs_6carrier_long_thermal_test_Aug2020/'
filename=os.path.join(filepath,f'{ctime0}_packets.log')

print(f'-> Logging to {filename}')

if not os.path.exists(filename):
    with open(filename,'a') as logf:
        logf.write(hdr)

while True:
    with open(filename,'a') as logf:
        ctime=int(time.time())
        for slot in slots:
            (frameCnt,frameSize,frameLossCnt,frameOutOrderCnt,badFrameCnt)=get_packet_stats(slot,timeout=10)
            logf.write(fmt.format([f'smurf_server_s{slot}',
                                   ctime,
                                   frameCnt,
                                   frameSize,
                                   frameLossCnt,
                                   frameOutOrderCnt,
                                   badFrameCnt])
            )
        time.sleep(cadence_sec)
