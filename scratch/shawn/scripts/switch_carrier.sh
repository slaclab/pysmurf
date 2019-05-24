#!/bin/bash
# [$1 == slot number to switch to]

slots_in_use=(4 5)
echo "slots_in_use = (${slots_in_use[@]})"

if [ $# -eq 0 ]; then
    echo "Must provide which slot you want to switch to."
    exit 1
fi

cpwd=$PWD
tmux_session_name=smurf

stop_pyrogue () {
    # stop pyrogue dockers, assumes the pyrogue version you want to use
    # has been soft linked to slotN/current
    echo "-> Stopping slot $1 pyrogue docker";
    cd /home/cryo/docker/smurf/slot$1/current;
    ./stop.sh;
}

if [[ " ${slots_in_use[*]} " != *" $1 "* ]]; then
    echo "asked for slot $1 : not in use (or not known to be in use)"
    exit
fi

# stop pyrogue servers on both carriers
#stop_pyrogue 4
#stop_pyrogue 5
#cd $cpwd

for s in "${slots_in_use[@]}"
do
    if [ "$s" -eq "$1" ]; then
	echo "switching to $s"
	# must shut down all other slots first
    else
	echo "shutting down $s"

	# stop pyrogue server
	stop_pyrogue $s
	sleep 2
	
	# stop tailing smurf_server_s4 log
	tmux select-window -t ${tmux_session_name}:$s
	tmux select-pane -t 0
	tmux send-keys -t ${tmux_session_name}:$s C-c		
    fi
done

# start the requested slot
slot_number=$1
tmux select-window -t ${tmux_session_name}:${slot_number}

tmux select-pane -t 0
tmux send-keys -t ${tmux_session_name}:${slot_number} C-c
tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd /home/cryo/docker/smurf/slot'${slot_number}'/current' C-m
tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m

tmux select-pane -t 1
