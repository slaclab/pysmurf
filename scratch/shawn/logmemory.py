import subprocess
import time
import os

ctime0=int(time.time())

cadence_sec=300

of = open(f'/data/smurf_data/server_memory_logs/{ctime0}_server_memory.log', "w")

while True:
    ctime=time.time()
    result = subprocess.run(['docker','container','stats','--no-stream','--no-trunc'], stdout=subprocess.PIPE)
    result=result.stdout.splitlines()
    for idx in range(len(result)):
        line=result[idx].decode('UTF-8')
        result[idx]=f'{ctime} : {line}'

        print(result[idx])
        of.write(f'{result[idx]}\n')
    
    of.flush()

    time.sleep(cadence_sec-1)
