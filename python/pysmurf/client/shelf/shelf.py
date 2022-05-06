"""
Interact with the SMuRF shelf manager, for example,
shm-smrf-sp01. The SMuRF shelf manager is its own host, accessible
from the SMuRF server only. Most common operation is deactivating and
activating one of the shelf's slots.
"""

import subprocess

def run_crate_command(command):
    crate_hostname = 'shm-smrf-sp01'
    subprocess.call(f'ssh {crate_hostname} {command}')

def reactivate_slot(slot):
    slot = str(slot)
    run_crate_command(f'clia deactivate board {slot}')
    time.sleep(5)
    run_crate_command(f'clia activate board {slot}')

