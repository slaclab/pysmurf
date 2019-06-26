#!/bin/bash

ctime=`date +%s`

attach_at_end=false
screenshot_signal_analyzer=false
configure_pysmurf=false
reboot=true
using_timing_master=false
run_half_band_test=false
one_at_a_time=true
write_config=false
cpwd=$PWD
pysmurf=/home/cryo/docker/pysmurf/dspv3
# path can assume you're in the pysmurf directory
pysmurf_init_script=scratch/shawn/scripts/init_nist_cmb.py

crate_id=3
slots_in_configure_order=(5)

tmux_session_name=smurf

matching_dockers () {
    # the $1 in the single quotes doesn't get replaced with the input
    # arg ; it's the first column of the grep output
    if [ -z "$(docker ps | grep $1 | awk '{print $1}')" ] ; then
	return 0
    else
	return 1
    fi
}

stop_pyrogue () {
    # stop pyrogue dockers, assumes the pyrogue version you want to use
    # has been soft linked to slotN/current
    echo "-> Stopping slot $1 pyrogue docker";
    cd /home/cryo/docker/smurf/slot$1/current;
    ./stop.sh;
}

# I hate this
wait_for_docker () {
    latest_docker=`docker ps -a -n 1 -q`    
}

start_slot_tmux () {
    slot_number=$1

    pysmurf_docker0=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
    
    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd /home/cryo/docker/smurf/slot'${slot_number}'/current' C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m


    echo '-> Waiting for smurf_server_s'${slot_number}' docker to start.'
    while [[ -z `docker ps  | grep smurf_server_s${slot_number}`  ]]; do
	sleep 1
    done
    echo '-> smurf_server_s'${slot_number}' docker started.'    
    
    echo '-> Waiting for smurf_server_s'${slot_number}' GUI to come up.'    
    sleep 2
    grep -q "Starting GUI" <(docker logs smurf_server_s${slot_number} -f)
    
    # start pysmurf in a split window and initialize the carrier
    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 1

    tmux send-keys -t ${tmux_session_name}:${slot_number} 'ipython3 -i '${pysmurf_init_script}' '${slot_number} C-m

    ## not the safest way to do this.  If someone else starts a
    ## pysmurf docker, will satisfy this condition.  Not even sure why
    ## this is even needed - why is there so much latency on
    ## smurf-srv03 between starting a docker and it showing up in
    ## docker ps?
    echo "pysmurf_docker0=$pysmurf_docker0"
    latest_pysmurf_docker=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
    echo "latest_pysmurf_docker=$latest_pysmurf_docker"	    
    while [ "$pysmurf_docker0" == "$latest_pysmurf_docker" ]; do
    	latest_pysmurf_docker=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
    	sleep 1
	echo "latest_pysmurf_docker=$latest_pysmurf_docker"		
    done
    
    # after running this, can run
    # pysmurf_docker=`docker ps -n 1 -q`
    # to hex of most recently created docker.
}

# right now, real dumb.  Assumes the active window in tmux is this
# slot's
config_pysmurf () {
    slot_number=$1
    pysmurf_docker=$2
    
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=True,make_logfile=False)' C-m
    
    # wait for setup to complete
    echo "-> Waiting for carrier setup (watching pysmurf docker ${pysmurf_docker})"
    # not clear why, but on smurf-srv03 need this wait or attempt to
    # wait until done with setup fails.
    sleep 2
    grep -q "Done with setup" <(docker logs $pysmurf_docker -f)
    echo "-> Carrier is configured"

    echo "-> Disable streaming (unless taking data)"
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_stream_enable(0)' C-m

    # write config
    if [ "$write_config" = true ] ; then
	sleep 2    
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_read_all(write_log=True); S.write_config("/home/cryo/shawn/'${ctime}'_slot'${slot_number}'.yml")' C-m
	sleep 45
    fi

    if [ "$run_half_band_test" = true ] ; then    
	sleep 2
	echo "-> Running half-band fill test"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'sys.argv[1]='${ctime}'; exec(open("scratch/shawn/half_band_filling_test.py").read())' C-m    
	grep -q "Done with half-band filling test." <(docker logs $pysmurf_docker -f)
    fi
    
    sleep 1
}

# use tmux instead
# https://www.peterdebelak.com/blog/tmux-scripting/

# kill smurf tmux, if running, and restart
echo "-> Killing ${tmux_session_name} tmux session"
tmux kill-session -t ${tmux_session_name}
echo "-> Starting a new ${tmux_session_name} tmux session"
tmux new-session -d -s ${tmux_session_name}
#tmux new -s ${tmux_session_name} -d

# stop pyrogue servers on all carriers
for slot in ${slots_in_configure_order[@]}; do
    stop_pyrogue $slot
done
cd $cpwd

# stop all pysmurf dockers
matching_dockers pysmurf
if [ "$?" = "1" ]; then
    echo "-> Stopping all running pysmurf dockers."
    docker rm -f $(docker ps | grep pysmurf | awk '{print $1}')
fi

# if using a timing master, check that timing docker is running,
# or else nothing will work.
if [ "$using_timing_master" = true ] ; then
    matching_dockers tpg_ioc
    if [ "$?" = "1" ]; then    
	echo "-> tpg_ioc docker is up."	
    else
	echo "-> tpg_ioc docker is down, must start."
	exit 1
    fi
fi

## will need a utils docker.  first remove all others to avoid
## proliferation, then start one in tmux
matching_dockers smurf-base
if [ "$?" = "1" ]; then
    echo "-> Stopping all running utils dockers."
    docker rm -f $(docker ps | grep smurf-base | awk '{print $1}')
fi

tmux rename-window -t ${tmux_session_name}:0 utils
tmux send-keys -t ${tmux_session_name}:0 'cd /home/cryo/docker/utils' C-m
tmux send-keys -t ${tmux_session_name}:0 './run.sh' C-m

# display tpg log in tmux 0 with utils term
tmux split-window -v -t ${tmux_session_name}:0
tmux send-keys -t ${tmux_session_name}:0 'docker logs tpg_ioc -f' C-m

# leave the utils pane selected
tmux select-window -t utils
tmux select-pane -t 0

if [ "$reboot" = true ] ; then

    # deactivate carriers
    echo "-> Deactivating carrier(s) ${slots_in_configure_order[@]}"
    for slot in ${slots_in_configure_order[@]}; do
	ssh root@shm-smrf-sp01 "clia deactivate board ${slot}"	
    done    
    
    echo "-> Waiting 5 sec before re-activating carrier(s)"
    sleep 5
    
    # reactivate carriers
    echo "-> Re-activating carrier(s) ${slots_in_configure_order[@]}"
    for slot in ${slots_in_configure_order[@]}; do
	ssh root@shm-smrf-sp01 "clia activate board ${slot}"	
    done        
fi

################################################################################
### Configure carriers

active_slot=
for slot in ${slots_in_configure_order[@]}; do
    # make sure ethernet is up on carrier
    echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
    cd $cpwd
    ping_carrier 10.0.${crate_id}.$((${slots_in_configure_order[0]}+100))

    # may only want one pyrogue server running at a time
    if [[ ! -z "$active_slot" && "$one_at_a_time" = true ]] ; then
	tmux select-window -t smurf_slot${active_slot}
	tmux select-pane -t 0
	tmux send-keys -t ${tmux_session_name}:${active_slot} C-c
	
	# stop smurf_server_s4
	stop_pyrogue ${active_slot}
    fi
    
    start_slot_tmux ${slot}

    pysmurf_docker_slot=`docker ps -a -n 1 -q`

    if [[ "$reboot" = true && "$configure_pysmurf" = true ]] ; then
    	config_pysmurf ${slot} ${pysmurf_docker_slot}
    fi
    
    active_slot=${slot}
done

echo "active_slot=${active_slot}"

### Done configuring carriers
################################################################################

if [ "$attach_at_end" = true ] ; then
    tmux attach -t ${tmux_session_name}
fi

# terminal running script that screenshots can't overlap with the
# remote desktop window, for some stupid reason.
if [ "$screenshot_signal_analyzer" = true ] ; then
    wid=`wmctrl -l | grep 171.64.108.28 | awk '{print $1}'`
    # bring to forefront
    wmctrl -a 171.64.108.28
    # screenshot
    import -window ${wid} /home/cryo/shawn/${ctime}_signal_analyzer.png
fi
    
