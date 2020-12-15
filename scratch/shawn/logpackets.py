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
    return (frameCntPV.get(timeout=timeout,as_string=True),
            frameSizePV.get(timeout=timeout,as_string=True),
            frameLossCntPV.get(timeout=timeout,as_string=True),
            frameOutOrderCntPV.get(timeout=timeout,as_string=True),
            badFrameCntPV.get(timeout=timeout,as_string=True),)

import time
ctime0=int(time.time())

fmt='{0[0]:<20}{0[1]:<20}{0[2]:<30}{0[3]:<30}{0[4]:<30}{0[5]:<30}{0[6]:<30}\n'

hdr=fmt.format(['epics_root','ctime','FrameCnt','FrameSize','FrameLossCnt','FrameOutOrderCnt','BadFrameCnt'])

filepath='/data/smurf_data/simonsobs_6carrier_long_thermal_test_Aug2020/'
filename=os.path.join(filepath,f'{ctime0}_packets.log')

print(f'-> Logging to {filename}')

if not os.path.exists(filename):
    with open(filename,'a') as logf:
        logf.write(hdr)

while True:
    with open(filename,'a') as logf:
        for slot in slots:
            ctime=int(time.time())            
            (frameCnt,frameSize,frameLossCnt,frameOutOrderCnt,badFrameCnt)=get_packet_stats(slot)

            entry=fmt.format([f'smurf_server_s{slot}',
                              ctime,
                              frameCnt,
                              frameSize,
                              frameLossCnt,
                              frameOutOrderCnt,
                              badFrameCnt])
            print(entry)
            logf.write(entry)
        time.sleep(cadence_sec)
