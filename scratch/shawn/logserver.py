import subprocess
import time

ctime0=int(time.time())

cadence_sec=60

of = open(f'/data/smurf_data/server_sensor_logs/{ctime0}_server_getsensorinfo.log', "w")

while True:
    ctime=time.time()
    result = subprocess.run(['racadm', 'getsensorinfo'], stdout=subprocess.PIPE)
    result=result.stdout.splitlines()
    for idx in range(len(result)):
        line=result[idx].decode('UTF-8')
        result[idx]=f'{ctime} : {line}'

        print(result[idx])
        of.write(f'{result[idx]}\n')
    
    of.flush()

    time.sleep(cadence_sec)
