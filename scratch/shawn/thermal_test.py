import time
import os
import fcntl

# Dumb monitoring of FPGA temperatures
# 1399  while true; do clear; awk '{print $1" "$3}' /data/smurf_data/20190719/1563578383/outputs/1563578385_hwlog.dat | tail -n 3; sleep 1; done


def tmux_cmd(slot_number,cmd,tmux_session_name='smurf'):
    os.system("""tmux send-keys -t {}:{} '{}' C-m""".format(tmux_session_name,slot_number,cmd))

def start_hardware_logging(slot_number,filename=None):
    cmd=None
    if filename is not None:
        cmd="""S.start_hardware_logging("{}")""".format(filename)
    else:
        cmd="""S.start_hardware_logging()"""
    tmux_cmd(slot_number,cmd)

def stop_hardware_logging(slot_number):
    cmd="""S.stop_hardware_logging()"""
    tmux_cmd(slot_number,cmd)    
    
def carrier_setup(slot_number):
    cmd="""S.setup()"""
    tmux_cmd(slot_number,cmd)       

def add_tag_to_hardware_log(hardware_logfile,tag):
    with open(hardware_logfile,'a') as logf:
        # file locking so multiple hardware loggers running in
        # multiple pysmurf sessions can write to the same
        # requested file if desired
        fcntl.flock(logf, fcntl.LOCK_EX)
        logf.write('#' + str(tag).rstrip() + '\n')
        fcntl.flock(logf, fcntl.LOCK_UN)
    
ctime=time.time()
output_dir='/data/smurf_data/rflab_thermal_testing_swh_July2019'
hardware_logfile=os.path.join(output_dir,'{}_hwlog.dat'.format(int(ctime)))

slots=[2]

wait_before_setup_min=0.1

# start hardware logging
for slot in slots:
    start_hardware_logging(slot,hardware_logfile)

print('-> Waiting {} min before setup.'.format(wait_before_setup_min))
wait_before_setup_sec=wait_before_setup_min*60
time.sleep(wait_before_setup_sec)
    
# setup
add_tag_to_hardware_log(hardware_logfile,tag='setup')
for slot in slots:      
    carrier_setup(slot)

# stop hardware logging
#for slot in slots:
#    stop_hardware_logging(slot)
