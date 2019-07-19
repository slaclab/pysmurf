#!/bin/bash

source shawnhammerfunctions

ctime=`date +%s`

set_crate_fans_to_full=true
attach_at_end=true
screenshot_signal_analyzer=false
configure_pysmurf=true
reboot=true
using_timing_master=false
run_half_band_test=false
one_at_a_time=true
write_config=false
start_atca_monitor=true
cpwd=$PWD

pysmurf=/home/cryo/docker/pysmurf/dspv3
shelfmanager=shm-smrf-sp01

crate_id=3
slots_in_configure_order=(4)

pysmurf_init_script=scratch/shawn/scripts/init_rflab.py

tmux_session_name=smurf

# If true, forces crate fans to full speed
if [ "$set_crate_fans_to_full" = true ] ; then
    ssh root@${shelfmanager} "clia minfanlevel 15; clia setfanlevel all 15"
    sleep 2
fi

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

if [ "$using_timing_master" = true ] ; then
    # display tpg log in tmux 0 with utils term
    tmux split-window -v -t ${tmux_session_name}:0
    tmux send-keys -t ${tmux_session_name}:0 'docker logs tpg_ioc -f' C-m
fi

if [ "$start_atca_monitor" = true ] ; then
    # stop any already running atca_monitor instances
    matching_dockers smurf-atca-monitor
    if [ "$?" = "1" ]; then
	echo "-> Stopping all running atca_monitor dockers."
	docker rm -f $(docker ps | grep atca_monitor | awk '{print $1}')
    fi
    # start the atca_monitor docker up in a pane of the utils session
    tmux split-window -v -t ${tmux_session_name}:0
    tmux send-keys -t ${tmux_session_name}:0 'cd /home/cryo/docker/atca_monitor' C-m
    tmux send-keys -t ${tmux_session_name}:0 './run.sh' C-m
fi

# leave the utils pane selected
tmux select-window -t utils
tmux select-pane -t 0

exit

if [ "$reboot" = true ] ; then

    # deactivate carriers
    echo "-> Deactivating carrier(s) ${slots_in_configure_order[@]}"
    for slot in ${slots_in_configure_order[@]}; do
	ssh root@${shelfmanager} "clia deactivate board ${slot}"	
    done    
    
    echo "-> Waiting 5 sec before re-activating carrier(s)"
    sleep 5
    
    # reactivate carriers
    echo "-> Re-activating carrier(s) ${slots_in_configure_order[@]}"
    for slot in ${slots_in_configure_order[@]}; do
	ssh root@${shelfmanager} "clia activate board ${slot}"	
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
    
