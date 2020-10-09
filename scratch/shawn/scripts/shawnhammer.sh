#!/bin/bash

# stalls sometimes for some reason...
#xhost +

#default startup cfg
startup_cfg=/data/smurf_startup_cfg/smurf_startup.cfg

#https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/
while getopts ":ic:t" opt; do
    case ${opt} in
      i )
	  # make sure this file exists
	  if [ ! -f $pysmurf/pysmurf/$OPTARG ]; then
              echo "Invalid Input: No file found at $pysmurf/pysmurf/$OPTARG" 1>&2
              exit 1	      
	      echo "File not found!"
	  fi
	  pysmurf_init_script=$OPTARG
          ;;
      c )
	  # make sure this file exists
	  if [ ! -f $OPTARG ]; then
              echo "Invalid Input: No file found at $OPTARG" 1>&2
              exit 1	      
	      echo "File not found!"
	  fi
	  startup_cfg=$OPTARG
          ;;      
      t )
	  run_thermal_test=true
	  ;;
      \? )
        echo "Invalid Option: -$OPTARG" 1>&2
        exit 1
        ;;
      : )
        echo "Invalid Option: -$OPTARG requires an argument" 1>&2
        exit 1
        ;;
    esac
done
shift $((OPTIND -1))

# can't hammer without a cfg ; exit if one doesn't exist
if [ ! -f "$startup_cfg" ]; then
    echo "$startup_cfg doesn't exist, unable to shawnhammer." 1>&2
    exit 1
fi
source ${startup_cfg}

# if enable-tmux-logging, check that the tmux-logging plugin is
# installed in /home/cryo/
if [[ "$enable_tmux_logging" = true && ! -d "/home/cryo/tmux-logging" ]] ; then
    echo "tmux logging enabled, but can't find tmux-logging plugin." 1>&2
    echo "To use this option, must install to /home/cryo/tmux-logging." 1>&2
    echo "Disabling tmux logging (enable_tmux_logging=false)." 1>&2
    enable_tmux_logging=false
fi

# must confirm a slot configuration has been provided.  If not
# then for backwards compatilibity, take current
if [ -z "$slot_cfgs" ]; then
    # if no slot_cfg defined in the startup cfg, may be providing
    # just a slot list and expecting all slots to point to
    # /home/docker/smurf/current.  That's true if user has
    # defined the slots_in_configure_order array.
    if [ -v "slots_in_configure_order" ]; then    
	echo "No slot_cfgs defined in $startup_cfg," 1>&2
	echo "but slots_in_configure_order is defined," 1>&2
	# Make sure a current softlink exists for the rogue docker
	# to be assigned to each slot listed in the slots_in_configure_order
	# array.
	if [ -L /home/cryo/docker/smurf/current ]; then
	    echo "Pointing every slot to /home/cryo/docker/smurf/current" 1>&2
	    
	    slots=("${slots_in_configure_order[@]}")
	    pyrogues=(`seq ${#slots_in_configure_order[@]} | awk '{print "/home/cryo/docker/smurf/current"}' | tr '\n' ' '`)
	else
	    echo "... but must define /home/cryo/docker/smurf/current." 1>&2
	    exit 1
	fi
    else
	echo "Neither slot_cfgs nor slots_in_configure_order defined in $startup_cfg." 1>&2
	exit 1
    fi
else
    if [ -v "slots_in_configure_order" ]; then
	echo "Both slot_cfgs and slots_in_configure_order defined in" 1>&2
	echo "$startup_cfg.  Must choose just one." 1>&2
	exit 1
    fi
    
    echo "Taking docker<->slot configuration from slot_cfgs." 1>&2    
    ## extract slot and configuration arrays
    # first column is the slot numbers, in slot configure order.
    slots=( $(awk '{print $1}' <<< "$slot_cfgs") )
    # second column is the pyrogue directories, in slot configure order    
    pyrogues=( $(awk '{print $2}' <<< "$slot_cfgs") )
    # third (optional) column is the pysmurf experiment.cfg
    pysmurf_cfgs=( $(awk '{print $3}' <<< "$slot_cfgs") )
fi

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
# Small wait to prevent https://github.com/slaclab/pysmurf/issues/115
sleep 0.1
echo "-> Starting a new ${tmux_session_name} tmux session"
tmux new-session -d -s ${tmux_session_name}
#tmux new -s ${tmux_session_name} -d

# if enable-tmux-logging, check that the tmux-logging plugin is
# installed in /home/cryo/
if [[ "$enable_tmux_logging" = true ]]; then
    mkdir -vp /data/smurf_data/tmux_logs
    tmux set -g @logging-path "/data/smurf_data/tmux_logs"
fi

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

# stop all smurf-streamer dockers
matching_dockers smurf-streamer
if [ "$?" = "1" ]; then
    echo "-> Stopping all running smurf-streamer dockers."
    docker rm -f $(docker ps -q -f name=smurf-streamer | awk '{print $1}')
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

# wait for utils docker to come up and record the container ID
utilsdockerid=`wait_for_docker_instance smurf-base`
docker rename ${utilsdockerid} smurf_utils

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
	    pysmurf_cfg=${pysmurf_cfgs[slot_idx]}

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
	    	echo "-> Waiting for server to come up on slot ${slot}."
	    	if is_slot_server_up ${slot}; then
	    	    slot_status[$slot_idx]=4;
	    	fi
	    fi
	    
	    # GUI is up.  Splits each slot window and instantiate
	    # pysmurf object
	    if [ "${slot_status[${slot_idx}]}" = "4" ]; then
	    	echo "-> Starting pysmurf on ${slot}."		
	    	start_slot_pysmurf ${slot} ${pysmurf_cfg}
	    	slot_status[$slot_idx]=5;
		if [ "${configure_pysmurf}" = false ]; then
		    # skip setup
	    	    slot_status[$slot_idx]=7;
		fi
	    fi

	    # Run pysmurf setup
	    if [ "${slot_status[${slot_idx}]}" = "5" ]; then
		echo "-> Running pysmurf setup on slot ${slot} ..."
		run_pysmurf_setup ${slot}
		slot_status[$slot_idx]=6
	    fi

	    # Check for pysmurf setup completion
	    if [ "${slot_status[${slot_idx}]}" = "6" ]; then
		echo "-> Waiting for carrier setup on slot ${slot} ..."		
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
	pysmurf_cfg=${pysmurf_cfgs[i]} 	

	# make sure ethernet is up on carrier
	echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
	cd $cpwd
	ping_carrier 10.0.${crate_id}.$((${slot}+100))
	
	start_slot_tmux_serial ${slot} ${pyrogue} ${pysmurf_cfg}
	
	pysmurf_docker_slot=`docker ps -a -n 1 -q`
	
	if [[ "$reboot" = true && "$configure_pysmurf" = true ]] ; then
    	    config_pysmurf_serial ${slot} ${pysmurf_docker_slot}
	fi
    done
fi

### Done configuring carriers
################################################################################

if [ "$run_thermal_test" = true ] ; then
    tmux new-window -t ${tmux_session_name}:8
    tmux rename-window -t ${tmux_session_name}:8 tests
    tmux send-keys -t ${tmux_session_name}:8 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:8 'ipython3 -i pysmurf/'${thermal_test_script}' '`echo ${slots[@]} | tr ' ' ,` C-m
fi

### Last thing ; script user can specify to run at the end of the
### hammer in each pysmurf session
if [ ! -z "$script_to_run" ]; then
    echo "-> Done hammering, running script_to_run=${script_to_run} on all slots ..."    
    for slot in ${slots[@]}; do    
	tmux send-keys -t ${tmux_session_name}:${slot} 'exec(open("'$script_to_run'").read())' C-m
    done
fi

if [ "$attach_at_end" = true ] ; then
    tmux attach -t ${tmux_session_name}
fi

# Rarely used for some lofi debugging at Stanford.
#
# terminal running script that screenshots can't overlap with the
# remote desktop window, for some stupid reason.
if [ "$screenshot_signal_analyzer" = true ] ; then
    wid=`wmctrl -l | grep 171.64.108.28 | awk '{print $1}'`
    # bring to forefront
    wmctrl -a 171.64.108.28
    # screenshot
    import -window ${wid} /home/cryo/shawn/${ctime}_signal_analyzer.png
fi
    
