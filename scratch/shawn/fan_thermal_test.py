import os

# be careful with this one
def disable_fan_control():
    cmd="""ssh root@shm-smrf-sp01 \"clia setfanpolicy 20 4 disable; clia setfanpolicy 20 3 disable\""""
    os.system(cmd)

def enable_fan_control():
    cmd="""ssh root@shm-smrf-sp01 \"clia setfanpolicy 20 4 enable; clia setfanpolicy 20 3 enable\""""
    os.system(cmd)    

def set_fan_level(fan_level=100):
    cmd="""ssh root@shm-smrf-sp01 \"clia minfanlevel %d; clia setfanlevel all %d\""""%(fan_level,fan_level)
    os.system(cmd)

ctime=time.time()
output_dir='/data/smurf_data/rflab_thermal_testing_swh_July2019'
hardware_logfile=os.path.join(output_dir,'{}_hwlog.dat'.format(int(ctime)))
    
slots=[2,3,4]
    
for slot in slots:
    start_hardware_logging(slot,hardware_logfile)

print('-> Waiting 1 min before changing fan levels.')
time.sleep(60)

fan_levels=range(70,101,2)[::-1]

for fan_level in fan_levels:
    print('-> Setting fan_level to %d'%fan_level)
    add_tag_to_hardware_log(hardware_logfile,tag='fan%d'%(fan_level))
    set_fan_level(fan_level)
    print('-> Waiting 5 min until next fan level.')
    time.sleep(5*60)
