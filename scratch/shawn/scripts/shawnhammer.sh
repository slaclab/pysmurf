#!/bin/bash

source shawnhammerfunctions

ctime=`date +%s`

shelfmanager=shm-smrf-sp01

set_crate_fans_to_full=true
# COMTEL max fan level is 100, ASIS is 15, ELMA is 15
## COMTEL in RF lab
#max_fan_level=100
## ELMA in RF lab
#max_fan_level=15
## ASIS in RF lab
max_fan_level=15

attach_at_end=true
screenshot_signal_analyzer=false
configure_pysmurf=true
reboot=true
using_timing_master=false
run_half_band_test=false
write_config=false
start_atca_monitor=true
# still not completely parallel.  Also doesn't work.
parallel_setup=false
cpwd=$PWD

pysmurf=/home/cryo/docker/pysmurf/dev

crate_id=1
slots_in_configure_order=(2)

pysmurf_init_script=scratch/shawn/scripts/init_ucsd.py

tmux_session_name=smurf

# If true, forces crate fans to full speed
if [ "$set_crate_fans_to_full" = true ] ; then
    ssh root@${shelfmanager} "clia minfanlevel ${max_fan_level}; clia setfanlevel all ${max_fan_level}"
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

if [ "$reboot" = true ] ; then

    # deactivate carriers
    deactivatecmd=""
    activatecmd=""    
    for slot in ${slots_in_configure_order[@]}; do
	deactivatecmd="$deactivatecmd clia deactivate board ${slot};"
	activatecmd="$activatecmd clia activate board ${slot};"	
    done

    # deactivate carriers
    echo "-> Deactivating carrier(s) ${slots_in_configure_order[@]}"    
    ssh root@${shelfmanager} "$deactivatecmd"
    
    echo "-> Waiting 5 sec before re-activating carrier(s)"
    sleep 5

    # activate carriers
    echo "-> Activating carrier(s) ${slots_in_configure_order[@]}"    
    ssh root@${shelfmanager} "$activatecmd"    
fi

################################################################################
### Configure carriers

if [ "$parallel_setup" = true ] ; then
    ## start parallel method
    # setup stages
    # 0 = carriers off.
    # 1 = carrier eth responds to ping.
    setup_complete=false
    completion_status=5
    declare -a slot_status=( $(for slot in ${slots_in_configure_order[@]}; do echo 0; done) )
    setup_loop_cadence_sec=1
    while [[ "${setup_complete}" = false ]] ; do 
	for slot_idx in `seq 0 $((${#slots_in_configure_order[@]}-1))`; do 
	    slot=${slots_in_configure_order[$slot_idx]}

	    if [ "${slot_status[${slot_idx}]}" = "0" ]; then
		# make sure ethernet is up on carrier
		echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
		cd $cpwdcase
		ping_carrier -q 10.0.${crate_id}.$((${slot}+100))
		# ping_carrier returns 0 if ping fails, 1 if it succeeds
		slot_status[$slot_idx]=$?
	    fi

	    if [ "${slot_status[${slot_idx}]}" = "1" ]; then
		echo "-> Creating tmux session and starting pyrogue on slot ${slot}."
		start_slot_tmux_and_pyrogue ${slot}
		slot_status[$slot_idx]=2
	    fi

	    if [ "${slot_status[${slot_idx}]}" = "2" ]; then
		echo "-> Waiting for pyrogue server to start on slot ${slot}."
		if is_slot_pyrogue_up ${slot}; then
		    slot_status[$slot_idx]=3;
		fi
	    fi

	    if [ "${slot_status[${slot_idx}]}" = "3" ]; then
		echo "-> Waiting for gui to come up on slot ${slot}."
		if is_slot_gui_up ${slot}; then
		    slot_status[$slot_idx]=4;
		fi
	    fi

	    # GUI is up.  Splits each slot window and instantiate
	    # pysmurf object
	    if [ "${slot_status[${slot_idx}]}" = "4" ]; then
		echo "-> Starting pysmurf on ${slot}."		
		start_slot_pysmurf ${slot}
		slot_status[$slot_idx]=5;
	    fi	    

	    ## STILL NEED PYSMURF INITIALIZATION AND CONFIGURE STAGES

	    # Check status
	    echo "slot_status="${slot_status[@]}
	    # check if complete
	    status_summary=(`echo ${slot_status[@]} | tr ' ' '\n' | sort | uniq`)
	    # break out of setup loop once all slot statuses reach completion status.
	    if [[ "${#status_summary[@]}" = "1" && "${status_summary[0]}" = "${completion_status}" ]] ; then
		setup_complete=true
	    fi
	done
	sleep ${setup_loop_cadence_sec}
    done
else
    ##  older serial method
    for slot in ${slots_in_configure_order[@]}; do
	# make sure ethernet is up on carrier
	echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
	cd $cpwd
	ping_carrier 10.0.${crate_id}.$((${slots_in_configure_order[0]}+100))
	
	start_slot_tmux_serial ${slot}
	
	pysmurf_docker_slot=`docker ps -a -n 1 -q`
	
	if [[ "$reboot" = true && "$configure_pysmurf" = true ]] ; then
    	    config_pysmurf_serial ${slot} ${pysmurf_docker_slot}
	fi
    done
fi

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
    
