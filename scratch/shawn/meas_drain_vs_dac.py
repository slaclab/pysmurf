import numpy as np
import os
import subprocess
import datetime
from datetime import datetime
import time

def tmux_cmd(slot_number,cmd,tmux_session_name='smurf'):
    os.system("""tmux send-keys -t {}:{} '{}' C-m""".format(tmux_session_name,slot_number,cmd))

def command_over_ssh(COMMAND,HOST="cryo@pc98970.slac.stanford.edu"):

    ssh = subprocess.Popen(["ssh", f"{HOST}", COMMAND],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        print >>sys.stderr, "ERROR: %s" % error
        return None
    else:
        return result

ctime=int(time.time())
print(f'ctime={ctime}')

carrier_slot=4
# 4K1
#dacno=31
# 4K2
#dacno=29
# 50K1
dacno=32
dac_voltages=np.linspace(-9.999,9.999,101)
wait_after_set_sec=1.

outfilename=f'{ctime}_drain_vs_dac{dacno}.dat'
of=open(outfilename,'w')

columns='vdac\tvdrain\n'
of.write(columns)

for v in dac_voltages:
    tmux_cmd(carrier_slot,f'S.set_rtm_slow_dac_volt({dacno},{v})')
    time.sleep(wait_after_set_sec)
    measv=float(command_over_ssh('python3 ~/ib_tester/meas_volt.py','cryo@pc98970.slac.stanford.edu')[0].rstrip())
    of.write(f'{v}\t{measv}\n')
    print(f'dac{dacno} voltage={v} V, measure {measv} V')

of.close()
