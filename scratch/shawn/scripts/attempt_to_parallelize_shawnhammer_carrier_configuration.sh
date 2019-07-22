crate_id=3
slots_in_configure_order=(2 3 4)

declare -a slot_status=( $(for slot in ${slots_in_configure_order[@]}; do echo 0; done) )
active_slot=
while true; do 
    for slot_idx in `seq 0 $((${#slots_in_configure_order[@]}-1))`; do
	slot=${slots_in_configure_order[$slot_idx]}

	if [ "${slot_status[${slot_idx}]}" = "0" ]; then
	    # make sure ethernet is up on carrier
	    echo "-> Waiting for ethernet on carrier in slot ${slot} to come up ..."
	    cd $cpwdcase
	    ping_carrier 10.0.${crate_id}.$((${slot}+100))
	    # ping_carrier returns 0 if ping fails, 1 if it succeeds
	    slot_status[$slot_idx]=$?
	fi
        
	#    # may only want one pyrogue server running at a time
	#    if [[ ! -z "$active_slot" && "$one_at_a_time" = true ]] ; then
	#	tmux select-window -t smurf_slot${active_slot}
	#	tmux select-pane -t 0
	#	tmux send-keys -t ${tmux_session_name}:${active_slot} C-c
	#	
	#	# stop smurf_server_s4
	#	stop_pyrogue ${active_slot}
	#    fi
	#    
	#    start_slot_tmux ${slot}
	#
	#    pysmurf_docker_slot=`docker ps -a -n 1 -q`
	#
	#    if [[ "$reboot" = true && "$configure_pysmurf" = true ]] ; then
	#    	config_pysmurf ${slot} ${pysmurf_docker_slot}
	#    fi
	#    
	#    active_slot=${slot}
    done
	
    echo "slot_status="${slot_status[@]}    
done
