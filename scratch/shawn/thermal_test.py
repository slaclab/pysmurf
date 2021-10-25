# Don't run inside pysmurf docker - run outside all dockers.
import time
import os
import fcntl
import sys
import epics

shelfmanager='shm-smrf-sp01'

def get_crate_mfr(shelfmanager,timeout=5):
    print(f'{shelfmanager}:Crate:Sensors:Crate:CrateInfo:manufacturer')
    crate_mfr=epics.caget(f'{shelfmanager}:Crate:Sensors:Crate:CrateInfo:manufacturer',as_string=True,timeout=5)
    return crate_mfr

# atca_monitor not working right now, have to hardcode.
crate_mfr='comtel'
#while crate_mfr is None:
#    print('Polling atca_monitor for crate manufacturer ...')
#    crate_mfr=get_crate_mfr(shelfmanager)
#
#print(f"Got crate manufacturer!  It's \"{crate_mfr}\"")
    
# Comtel 7-slot crate defaults
max_fan_level=100
fan_frus=[(20,4),(20,3)]
restricted_fan_level=50
if 'comtel' in crate_mfr.lower():
    # Comtel 7-slot settings are default values above.
    print('COMTEL crate!')
    pass
elif 'asis' in crate_mfr.lower():
    # ASIS 7-slot
    print('ASIS crate!  Adjusting fan levels and FRU ids.')
    max_fan_level=15
    fan_frus=[(20,10),(20,9)]
    restricted_fan_level=7
else:
    print(f'Thermal test not supported for crate manufacturer "{crate_mfr}".')
    print(f'Aborting thermal test!')
    sys.exit(1)

# Takes a comma delimited list of slots to run the thermal test on
slots=[int(slot) for slot in sys.argv[1].split(',')]
print(f'Running thermal test on slots {slots} ...')

# whether or not to automatically plot the hardware logs
gnuplot_temperatures=True
set_fans_to_full_at_start=True

wait_to_check_full_band_response=False
stop_logging_at_end=False
log_temperatures=True
pause_btw_stages=False
pause_btw_band_fills=False
pause_btw_eta_scans=False
skip_setup=False
bands=range(8)

reset_rate_khz=10
measure_full_band_response_after_setup=True
wait_before_full_band_response_at_end_min=1
measure_full_band_response_at_end=True

wait_before_setup_min=1
wait_after_setup_min=1
wait_btw_band_fills_min=1
wait_after_band_fills_min=1

# wait btw slots if using the ethernet interface to keep from stalling
wait_btw_eta_scans_min=0
wait_after_eta_scans_min=0
wait_btw_slot_eta_scans_min=1

wait_after_streaming_on_min=1
# this wait is important because there's no easy way to know when each
# bands' tracking setup is complete.
wait_btw_tracking_setups_min=1
wait_after_tracking_setups_min=1

full_fan_level_dwell_min=2

# Whether or not to restrict the fan level
restrict_fan_level=True
restricted_fan_level_dwell_min=15

# Dumb monitoring of FPGA and regulator temperatures
#while true; do clear; awk '{print $1" "$3" "$17" "$18" "$19}' 1568751172_hwlog.dat | tail -n 3 | sort; sleep 1; done

def tmux_cmd(slot_number,cmd,tmux_session_name='smurf'):
    print(slot_number, cmd, tmux_session_name)
    os.system("""tmux send-keys -t {}:{} '{}' C-m""".format(tmux_session_name,slot_number,cmd))

def get_eta_scan_in_progress(slot_number,band,tmux_session_name='smurf',timeout=5):
    etaScanInProgress=epics.caget('smurf_server_s{}:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base[{}]:CryoChannels:etaScanInProgress'.format(slot_number,band),timeout=timeout)

    if type(etaScanInProgress) == type(None):
        print("Failed to caget etaScanInProgress. Trying again.")
        return True

    print("Got etaScanInProgress. Value is", etaScanInProgress)

    return int(etaScanInProgress)
    
def start_hardware_logging(slot_number,filename=None):
    cmd=None
    if filename is not None:
        cmd="""S.start_hardware_logging("{}")""".format(filename)
    else:
        cmd="""S.start_hardware_logging()"""
    tmux_cmd(slot_number,cmd)

# has to be done on a slot, but doesn't matter which
def amcc_dump(slot_number,fpath,shelfmanager='shm-smrf-sp01'):
    cmd="""os.system("amcc_dump --all %s > %s")"""%(shelfmanager,fpath)
    tmux_cmd(slot_number,cmd)

# has to be done on a slot, but doesn't matter which    
def amcc_dump_bsi(slot_number,fpath,shelfmanager='shm-smrf-sp01'):
    cmd="""os.system("amcc_dump_bsi --all %s > %s")"""%(shelfmanager,fpath)
    tmux_cmd(slot_number,cmd)    
    
def stop_hardware_logging(slot_number):
    cmd="""S.stop_hardware_logging()"""
    tmux_cmd(slot_number,cmd)    

def disable_streaming(slot_number):
    print("-> Disable streaming")
    cmd="""S.set_stream_enable(0)"""
    tmux_cmd(slot_number,cmd)

def enable_streaming(slot_number):
    print("-> Enable streaming")
    cmd="""S.set_stream_enable(1)"""
    tmux_cmd(slot_number,cmd)    

def carrier_setup(slot_number,shelfmanager='shm-smrf-sp01'):
    cmd="""S.shelf_manager=\"%s\"; S.setup()"""%(shelfmanager)
    tmux_cmd(slot_number,cmd)

def write_carrier_state(slot_number,filename):
    cmd="""S.write_state(\"{}\")""".format(filename)
    tmux_cmd(slot_number,cmd)

def write_atca_monitor_state(slot_number,filename):
    cmd="""S.write_atca_monitor_state(\"{}\")""".format(filename)
    tmux_cmd(slot_number,cmd)        

def extend_argv_if_needed(slot_number):
    cmd="""if len(sys.argv)==1: sys.argv=[sys.argv[0],None]"""
    tmux_cmd(slot_number,cmd)
    
def measure_full_band_response(slot_number):
    cmd="""exec(open("/usr/local/src/pysmurf/scratch/shawn/full_band_response.py").read())"""
    tmux_cmd(slot_number,cmd)    
    
def fill_band(slot_number,band):
    cmd="""sys.argv[1]={}; exec(open("/usr/local/src/pysmurf/scratch/shawn/fill_band.py").read())""".format(band)
    tmux_cmd(slot_number,cmd)    

def add_tag_to_hardware_log(hardware_logfile,tag):
    with open(hardware_logfile,'a') as logf:
        # file locking so multiple hardware loggers running in
        # multiple pysmurf sessions can write to the same
        # requested file if desired
        fcntl.flock(logf, fcntl.LOCK_EX)
        logf.write('#' + str(tag).rstrip() + '\n')
        fcntl.flock(logf, fcntl.LOCK_UN)

def eta_scan_band(slot_number,band):
    cmd="""S.run_serial_eta_scan({})""".format(band)
    tmux_cmd(slot_number,cmd)

def tracking_setup_band(slot_number,band,reset_rate_khz):
    cmd="""S.tracking_setup({},reset_rate_khz={})""".format(band,reset_rate_khz)
    tmux_cmd(slot_number,cmd)    

def get_last_line_tmux(slot_number,tmux_session_name='smurf',offset=0):
    import subprocess
    from subprocess import Popen
    p1 = subprocess.Popen(['tmux','capture-pane','-pt','{}:{}'.format(tmux_session_name,slot_number)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['tail','-n','%d'%(1-offset)], stdin=p1.stdout, stdout=subprocess.PIPE)
    result=p2.communicate()[0].decode('UTF-8')
    return result.split('\n')[0]
    
def wait_for_text_in_tmux(slot_number,text,tmux_session_name='smurf'):
    ret=1
    while ret:
        import subprocess
        from subprocess import Popen
        p1 = subprocess.Popen(['tmux','capture-pane','-pt','{}:{}'.format(tmux_session_name,slot_number)], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['grep','-q',text], stdin=p1.stdout, stdout=subprocess.PIPE)
        p2.communicate()
        ret=p2.returncode

# this sets them all at once
#def set_fan_level(shelfmanager,fanlevel):
#    os.system(f'ssh root@{shelfmanager} \"clia minfanlevel {fanlevel}; clia setfanlevel all {fanlevel}\"')
# this sets them by fru
def set_fan_level(shelfmanager,frus,fan_level,wait_btw_sec=1):
    command=''
    for fru1,fru2 in frus:
        command+=f'clia minfanlevel {fru1} {fru2} {fan_level}; sleep {wait_btw_sec}; clia setfanlevel {fru1} {fru2} {fan_level}; sleep {wait_btw_sec};'
    print(command)
    os.system(f'ssh root@{shelfmanager} \"{command}\"')

def disable_fan_policy(shelfmanager,frus,wait_btw_sec=1):
    command=''
    for fru1,fru2 in frus:
        command+=f'clia setfanpolicy {fru1} {fru2} DISABLE; sleep {wait_btw_sec};'
    print(command)
    os.system(f'ssh root@{shelfmanager} \"{command}\"')

def enable_fan_policy(shelfmanager,frus,wait_btw_sec=1):
    command=''
    for fru1,fru2 in frus:
        command+=f'clia setfanpolicy {fru1} {fru2} ENABLE; sleep {wait_btw_sec};'
    print(command)
    os.system(f'ssh root@{shelfmanager} \"{command}\"')

def record_hardware_state(slots,atca_yml,server_ymls,amcc_dump_file,shelfmanager,amcc_dump_bsi_file):
    # log atca and server ymls
    print('-> Writing ATCA state to {}.'.format(atca_yml))
    write_atca_monitor_state(slots[0],atca_yml)
    
    # write rogue server state
    for slot in slots:
        write_carrier_state(slot,server_ymls.format(slot))

    # NOT WORKING, NOT SURE WHY
    # only need once
    print('-> Writing output of amcc_dump to {}.'.format(amcc_dump_file))
    amcc_dump(slots[0],amcc_dump_file,shelfmanager)

    # NOT WORKING, NOT SURE WHY
    print('-> Writing output of amcc_dump_bsi to {}.'.format(amcc_dump_bsi_file))
    amcc_dump_bsi(slots[0],amcc_dump_bsi_file,shelfmanager)

    for slot in slots:
        cmd=f'docker logs smurf_server_s{slot} > {output_dir}/{ctime}_s{slot}dockerlog.dat'
        os.system(cmd)

ctime=time.time()
#output_dir='/data/smurf_data/westpak_thermal_testing_Jan2020'
#output_dir='/data/smurf_data/simonsobs_first10carriers_thermal_testing_Feb2020'
output_dir='/data/smurf_data/simonsobs_6carrier_long_thermal_test_Aug2020'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
hardware_logfile=os.path.join(output_dir,'{}_hwlog.dat'.format(int(ctime)))
atca_yml=os.path.join(output_dir,'{}_atca.yml'.format(int(ctime)))
server_ymls=os.path.join(output_dir,'{}'.format(int(ctime))+'_s{}.yml')
amcc_dump_file=os.path.join(output_dir,'{}'.format(int(ctime))+'_amcc_dump.txt')
amcc_dump_bsi_file=os.path.join(output_dir,'{}'.format(int(ctime))+'_amcc_dump_bsi.txt')

if set_fans_to_full_at_start:
    print(f'-> Setting fan speeds to full (={max_fan_level})')
    set_fan_level(shelfmanager,fan_frus,max_fan_level)
    time.sleep(5)

# start hardware logging
if log_temperatures:
    print('-> Logging to {}.'.format(hardware_logfile))    
    for slot in slots:
        start_hardware_logging(slot,hardware_logfile)

# STUPID HACK - make sure sys.argv is big enough in all the pysmurf
# sessions so that we can use it to pass args to open().read() calls.
for slot in slots:
    extend_argv_if_needed(slot)
    
if not skip_setup:

    if gnuplot_temperatures:
        while not os.path.exists(hardware_logfile):
            print(f'Waiting for {hardware_logfile} to start being populated ...')
            time.sleep(5)
        plot_cmd='cd ./pysmurf/scratch/shawn/; ./plot_temperatures.sh %s %s'%(','.join([str(slot) for slot in slots]),hardware_logfile)
        os.system(plot_cmd)

    print('-> Waiting {} min before setup.'.format(wait_before_setup_min))
    wait_before_setup_sec=wait_before_setup_min*60
    time.sleep(wait_before_setup_sec)    
    
    # setup
    add_tag_to_hardware_log(hardware_logfile,tag='setup')
    for slot in slots:      
        carrier_setup(slot,shelfmanager)

    print('-> Waiting for setup(s) to complete.')    
    for slot in slots:
        wait_for_text_in_tmux(slot,"Done with setup")
        
    # Disable streaming
    print('-> Disabling streaming')
    for slot in slots:
        disable_streaming(slot)
    
    print('-> Waiting {} min after setup.'.format(wait_after_setup_min))
    wait_after_setup_sec=wait_after_setup_min*60
    time.sleep(wait_after_setup_sec)

    if pause_btw_stages:
        input('Press enter to continue ...')
else:
    print('-> Waiting {} min instead of setup.'.format(wait_before_setup_min))
    wait_before_setup_sec=wait_before_setup_min*60
    time.sleep(wait_before_setup_sec)        

if measure_full_band_response_after_setup:
    for slot in slots:
        print(f'-> Checking full band response to confirm RF is properly configured on slot {slot}.')    
        measure_full_band_response(slot)
        wait_for_text_in_tmux(slot,"Done running full_band_response.py.")
        if wait_to_check_full_band_response:
            input(f'-> Visually check the measured full band response on slot {slot} before continuing (press enter)...')
        else:
            print(f'-> Done with full band response on slot {slot}.')
    
# fill bands, one at a time
wait_btw_band_fills_sec=wait_btw_band_fills_min*60
for band in bands:
    add_tag_to_hardware_log(hardware_logfile,tag='b{}fill'.format(band))        
    for slot in slots:
        fill_band(slot,band)
    print('-> Waiting {} min after band {} fill.'.format(wait_btw_band_fills_min,band))
    time.sleep(wait_btw_band_fills_sec)
    if pause_btw_band_fills:
        input('Press enter to continue ...')    

print('-> Waiting {} min after band fills.'.format(wait_after_band_fills_min))
wait_after_band_fills_sec=wait_after_band_fills_min*60
time.sleep(wait_after_band_fills_sec)

if pause_btw_stages:
    input('Press enter to continue ...')

# eta scan
wait_btw_eta_scans_sec=wait_btw_eta_scans_min*60
for band in bands:
    add_tag_to_hardware_log(hardware_logfile,tag='b{}eta'.format(band))        
    for slot in slots:
        print('-> Running eta scan on slot {}, band {}...'.format(slot,band))        
        eta_scan_band(slot,band)
    time.sleep(1)
    #wait for eta scans to complete
    for slot in slots:
        # Code breaks here.
        # Assume we don't actually need to wait for the eta scan to finish.
        #while get_eta_scan_in_progress(slot,band):
        #    time.sleep(5)
        print('-> Eta scan for slot {}, band {} completed.'.format(slot,band))

        # only need this if using eth interface
        wait_btw_slot_eta_scans_sec=wait_btw_slot_eta_scans_min*60                
        print('-> Waiting {} min btw eta scans on different slots.'.format(wait_btw_slot_eta_scans_min))
        time.sleep(wait_btw_slot_eta_scans_sec)        
        
    print('-> All band {} eta scans completed.'.format(band))
    print('-> Waiting {} min after band {} eta scans.'.format(wait_btw_eta_scans_min,band))            
    time.sleep(wait_btw_eta_scans_sec)
    if pause_btw_eta_scans:
        input('Press enter to continue ...')            

print('-> Waiting {} min after eta scans.'.format(wait_after_eta_scans_min))
wait_after_eta_scans_sec=wait_after_eta_scans_min*60
time.sleep(wait_after_eta_scans_sec)

if pause_btw_stages:
    input('Press enter to continue ...')

#######################################################################    
# tracking setup on all bands
wait_btw_tracking_setups_sec=wait_btw_tracking_setups_min*60
for band in bands:
    add_tag_to_hardware_log(hardware_logfile,tag='b{}tracking_setup'.format(band))        
    for slot in slots:
        print('-> Running tracking setup on slot {}, band {}...'.format(slot,band))        
        tracking_setup_band(slot,band,reset_rate_khz)
        print('-> Waiting {} min after band {} tracking setup.'.format(wait_btw_eta_scans_min,band))
        time.sleep(wait_btw_tracking_setups_sec)

print('-> Tracking setup run on all bands on slot {}.'.format(slot))
        
print('-> Waiting {} min after tracking setups.'.format(wait_after_tracking_setups_min))
wait_after_tracking_setups_sec=wait_after_tracking_setups_min*60
time.sleep(wait_after_tracking_setups_sec)
###### done running tracking setup

if pause_btw_stages:
    input('Press enter to continue ...')

#######################################################################    
# streaming on
for slot in slots:
    print('-> Enabling streaming on slot {}.'.format(slot))
    enable_streaming(slot)
add_tag_to_hardware_log(hardware_logfile,tag='streaming'.format(band))            

print('-> Waiting {} min after turning on streaming for all slots.'.format(wait_after_streaming_on_min))
wait_after_streaming_on_sec=wait_after_streaming_on_min*60
time.sleep(wait_after_streaming_on_sec)
###### done enabling streaming

if pause_btw_stages:
    input('Press enter to continue ...')        

##### DWELL WITH EVERYTHING ON AND FANS AT FULL LEVEL
full_fan_level_dwell_sec=full_fan_level_dwell_min*60
add_tag_to_hardware_log(hardware_logfile,tag='startfullfandwell')        
print('-> Dwelling for {} min with everything on at full fan level ...'.format(full_fan_level_dwell_min))
time.sleep(full_fan_level_dwell_sec)
add_tag_to_hardware_log(hardware_logfile,tag='endfullfandwell')

if restrict_fan_level:
    print(f'-> Restricting fan speeds to {restricted_fan_level} (out of {max_fan_level}).')
    time.sleep(5)

    restricted_fan_level_dwell_sec=restricted_fan_level_dwell_min*60.

    print('-> Dwelling for {} min at restricted fan level ...'.format(restricted_fan_level_dwell_min))    
    add_tag_to_hardware_log(hardware_logfile,tag='startrestrictfandwell')    

    start_restrict_fan_time=time.time()
    # must repeatedly disable the fan policy, for some reason
    while True:
        disable_fan_policy(shelfmanager,fan_frus)
        set_fan_level(shelfmanager,fan_frus,restricted_fan_level)
        # only do it for the dwell time
        if (time.time()-start_restrict_fan_time)>restricted_fan_level_dwell_sec:
            # record hardware state after dwelling at restricted fan speed
            record_hardware_state(slots,atca_yml,server_ymls,amcc_dump_file,shelfmanager,amcc_dump_bsi_file)
            break
        # wait 15 sec between hammering fan policy
        time.sleep(15)
        
    add_tag_to_hardware_log(hardware_logfile,tag='endrestrictfandwell')        

# done restricting fan ; re-enable fan policy
print(f'-> Done restricting fan speeds, re-enabling the fan policy...')
enable_fan_policy(shelfmanager,fan_frus)

############################################################
## Measure full band response at the end
#if measure_full_band_response_at_end:
#    print('-> Waiting {} min before full band response at end.'.format(wait_before_full_band_response_at_end_min))
#    #wait_before_full_band_response_at_end_sec=wait_before_full_band_response_at_end_min*60
    #time.sleep(wait_before_full_band_response_at_end_sec)    
#    time.sleep(60)
    
#    for slot in slots:
#        print(f'-> Checking full band response to confirm RF is still properly configured on slot {slot}.')    
#        measure_full_band_response(slot)
#        wait_for_text_in_tmux(slot,"Done running full_band_response.py.")
#        print(f'-> Done with full band response on slot {slot}.')

# stop hardware logging
if stop_logging_at_end:
    if log_temperatures:
        for slot in slots:
            stop_hardware_logging(slot)
else:
    print('Still logging ...')

# plot
#os.system(f'gnuplot -p -c pysmurf/scratch/shawn/plot_temperatures.gnuplot {hardware_logfile}')
print(f'gnuplot -p -c pysmurf/scratch/shawn/plot_temperatures.gnuplot {hardware_logfile}')
