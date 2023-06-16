import subprocess
import re
import sys

result = subprocess.run(['ssh', 'root@shm-smrf-sp01','clia sensordata'], stdout=subprocess.PIPE)
sensordata=result.stdout

tsensors={}
for sensor in sensordata.split(b'\n\n')[1:]:
    sn=None
    ipmbaddr=None
    for idx,field in enumerate(sensor.split(b'\n')):
        if len(field)==0:
            continue
        if idx==0:
            x = re.search(b"\(\"([^\)]+)\"\)",field)
            sn=x[1]
            ipmbaddr=field.split(b':')[0]
            if sn not in tsensors:
                tsensors[sn]={}
            ipmbaddr=field.split(b':')[0]
            tsensors[sn][ipmbaddr]={}
        if b'Type:' in field:
            x = re.search(b"\"(.*?)\"",field)
            tsensors[sn][ipmbaddr]['Type']=x[1]
        if b'Processed data:' in field:
            pd=field.split(b':')[-1:][0]
            tsensors[sn][ipmbaddr]['Processed data']=pd.split()[0]
            tsensors[sn][ipmbaddr]['Processed data units']=b' '.join(pd.split()[1:])

# Get thresholds
result = subprocess.run(['ssh', 'root@shm-smrf-sp01','clia getthreshold'], stdout=subprocess.PIPE)
thresholds=result.stdout

for threshold in thresholds.split(b'\n\n')[1:]:
    sn=None
    for idx,field in enumerate(threshold.split(b'\n')):        
        if len(field)==0:
            continue
        if idx==0:
            x = re.search(b"\(\"([^\)]+)\"\)",field)
            sn=x[1]
            ipmbaddr=field.split(b':')[0]
        if sn is not None and sn not in tsensors.keys():
            continue
        if sn is not None and ipmbaddr not in tsensors[sn].keys():
            continue
        for t in [b'Upper Non-Critical Threshold']:
            if t in field:
                pd=field.split(b':')[-1:][0]
                tsensors[sn][ipmbaddr][t]=pd.split()[0]
                tsensors[sn][ipmbaddr][t+b' units']=b' '.join(pd.split()[1:])

# Check vs thresholds!
for sn in tsensors.keys():
    for ipmbaddr in tsensors[sn].keys():
        ts=tsensors[sn][ipmbaddr]
        if ( ts['Type']==b'Temperature' and
             b'Upper Non-Critical Threshold' in ts.keys() and
             'Processed data' in ts.keys() ):
            unct=ts[b'Upper Non-Critical Threshold']
            pd=ts['Processed data']
            if float(pd) < float(unct):
                print(f'\033[92m{ipmbaddr.decode("utf-8")} : {sn.decode("utf-8")} : {float(pd):.3f} < {float(unct):.3f}\033[00m')
            else:
                print(f'\033[91m{ipmbaddr.decode("utf-8")} : {sn.decode("utf-8")} : {float(pd):.3f} > {float(unct):.3f}\033[00m')
    
