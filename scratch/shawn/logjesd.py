from epics import PV
import os
import sys

acqtime_sec=int(sys.argv[1])
slots=[4]
cadence_sec=5

pvs = {}
keys = []
for slot in slots:
    pvs[slot]={}
    for bay in [0,1]:
        pvs[slot][f'jesdtxdvb{bay}'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdTx:DataValid')
        keys.append(f'jesdtxdvb{bay}')
        pvs[slot][f'jesdrxdvb{bay}'] = PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdRx:DataValid')
        keys.append(f'jesdrxdvb{bay}')
        
        for cnt in range(10):
            pvs[slot][f'jesdrxsvb{bay}cnt{cnt}']=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdRx:StatusValidCnt[{cnt}]')
            keys.append(f'jesdrxsvb{bay}cnt{cnt}')
            pvs[slot][f'jesdtxsvb{bay}cnt{cnt}']=PV(f'smurf_server_s{slot}:AMCc:FpgaTopLevel:AppTop:AppTopJesd[{bay}]:JesdTx:StatusValidCnt[{cnt}]')
            keys.append(f'jesdtxsvb{bay}cnt{cnt}')

def get_jesd_stats(pvs,timeout=5):
    result = {}
    for k in pvs.keys():
        result[k]=pvs[k].get()
    return result

import time
ctime0=int(time.time())

fmt='{0[0]:<20}{0[1]:<20}'
k0=list(pvs.keys())[0]
for ii in range(len(pvs[k0])):
    fmt += '{0['+str(ii+2)+']:<20}'
fmt+='\n'

hdr=fmt.format(['epics_root',
                'ctime'] + keys)

filepath='/data/smurf_data/'
filename=os.path.join(filepath,f'{ctime0}_jesdstats.log')

print(f'-> Logging to {filename}')

if not os.path.exists(filename):
    with open(filename,'a') as logf:
        logf.write(hdr)

t0=time.time()
while time.time()-t0 < acqtime_sec:
    with open(filename,'a') as logf:
        for slot in slots:
            ctime=int(time.time())

            try:
                result=get_jesd_stats(pvs[slot])
                entry=fmt.format([f'smurf_server_s{slot}',
                                  ctime] + [result[k] for k in keys])
                #print(entry)
                logf.write(entry)                
            except:
                print("Cagets stalled!")
                
        time.sleep(cadence_sec)
