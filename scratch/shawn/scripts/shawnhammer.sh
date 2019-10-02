#!/bin/bash

startup_cfg=/data/smurf_startup_cfg/smurf_startup.cfg
if [ ! -f "$startup_cfg" ]; then
    echo "$startup_cfg doesn't exist, unable to shawnhammer."
    exit 1
fi

# didn't exit, so there must be a startup cfg file.  Load it.
source ${startup_cfg}

## extract slot and configuration arrays
# first column is the slot numbers, in slot configure order.
slots=( $(awk '{print $1}' <<< "$slot_cfgs") )
# second column is the pyrogue directories, in slot configure order
pyrogues=( $(awk '{print $2}' <<< "$slot_cfgs") )

source shawnhammerfunctions

ctime=`date +%s`

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
for ((i=0; i<${#slots[@]}; ++i)); do
    slot=${slots[i]}
    pyrogue=${pyrogues[i]} 
    stop_pyrogue $slot $pyrogue
done
cd $cpwd

# stop all pysmurf dockers
matching_dockers pysmurf
if [ "$?" = "1" ]; then
    echo "-> Stopping all running stable pysmurf dockers."
    # this stops all pysmurf dockers
    #docker rm -f $(docker ps | grep pysmurf | awk '{print $1}')
    docker rm -f $(docker ps -q -f name=pysmurf | awk '{print $1}')
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
    for slot in ${slots[@]}; do
	deactivatecmd="$deactivatecmd clia deactivate board ${slot};"
	activatecmd="$activatecmd clia activate board ${slot};"	
    done

    # deactivate carriers
    echo "-> Deactivating carrier(s) ${slots[@]}"    
    ssh root@${shelfmanager} "$deactivatecmd"
    
    echo "-> Waiting 5 sec before re-activating carrier(s)"
    sleep 5

    # activate carriers
    echo "-> Activating carrier(s) ${slots[@]}"    
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
    completion_status=7
    declare -a slot_status=( $(for slot in ${slots[@]}; do echo 0; done) )
    setup_loop_cadence_sec=1
    while [[ "${setup_complete}" = false ]] ; do
	for ((slot_idx=0; slot_idx<${#slots[@]}; ++slot_idx)); do
	    slot=${slots[slot_idx]}
	    pyrogue=${pyrogues[slot_idx]} 	

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
		start_slot_tmux_and_pyrogue ${slot} ${pyrogue}
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

	    # Run pysmurf setup
	    if [ "${slot_status[${slot_idx}]}" = "5" ]; then
		echo "-> Running pysmurf setup on slot ${slot}."
		run_pysmurf_setup ${slot}
		slot_status[$slot_idx]=6
	    fi

	    # Check for pysmurf setup completion
	    if [ "${slot_status[${slot_idx}]}" = "6" ]; then
		echo "-> Waiting for carrier setup on slot ${slot} (watching pysmurf docker ${pysmurf_docker})"		
	    	if is_slot_pysmurf_setup_complete ${slot}; then
	    	    slot_status[$slot_idx]=7;
	    	fi		
	    fi	    	    

	    # check if complete
	    status_summary=(`echo ${slot_status[@]} | tr ' ' '\n' | sort | uniq`)
	    # break out of setup loop once all slot statuses reach completion status.
	    if [[ "${#status_summary[@]}" = "1" && "${status_summary[0]}" = "${completion_status}" ]] ; then
		setup_complete=true
	    fi
	done

	# Print status
	echo "slot_status="${slot_status[@]}

	# Wait requested cadence between setup steps.
	sleep ${setup_loop_cadence_sec}
    done
else
    ##  older serial method
    for ((i=0; i<${#slots[@]}; ++i)); do
	slot=${slots[i]}
	pyrogue=${pyrogues[i]} 

	# make sure ethernet is up on carrier
	echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
	cd $cpwd
	ping_carrier 10.0.${crate_id}.$((${slot}+100))
	
	start_slot_tmux_serial ${slot} ${pyrogue}
	
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
    
