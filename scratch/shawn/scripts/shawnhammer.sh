#!/bin/bash

reboot=true
configure_hb=true
cpwd=$PWD

tmux_session_name=smurf

stop_pyrogue () {
    # stop pyrogue dockers, assumes the pyrogue version you want to use
    # has been soft linked to slotN/current
    echo "-> Stopping slot $1 pyrogue docker";
    cd /home/cryo/docker/smurf/slot$1/current;
    ./stop.sh;
}

start_slot_tmux () {
    slot_number=$1

    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd /home/cryo/docker/smurf/slot'${slot_number}'/current' C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m

    # might be nicer to wait for heartbeat, like smurf_server_s${slot_number}:AMCc:Time
    sleep 2
    grep -q "Starting GUI" <(docker logs smurf_server_s${slot_number} -f)
    
    # start pysmurf in a split window and initialize the carrier
    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd /home/cryo/docker/pysmurf/dspv3' C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 1

    tmux send-keys -t ${tmux_session_name}:${slot_number} 'ipython3 -i scratch/eyy/ipython_start/init_dspv3.py '${slot_number} C-m
    # after running this, can run
    # pysmurf_docker=`docker ps -n 1 -q`
    # to hex of most recently created docker.
}

# right now, real dumb.  Assumes the active window in tmux is this slot's
config_pysmurf () {
    slot_number=$1
    pysmurf_docker=$2
    
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=True,make_logfile=False)' C-m
    
    # wait for setup to complete
    echo "-> Waiting for carrier setup"
    grep -q "Done with setup" <(docker logs $pysmurf_docker -f)
    echo "-> Carrier is configured"

    echo "-> Disable streaming (unless taking data)"
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_stream_enable(0)' C-m
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

# stop pyrogue servers on both carriers
stop_pyrogue 4
stop_pyrogue 5
cd $cpwd

# stop all pysmurf dockers
echo "-> Stopping all running pysmurf dockers"
docker rm -f $(docker ps | grep pysmurf | awk '{print $1}')

## will need a utils docker.  first remove all others to avoid
## proliferation, then start one in tmux
echo "-> Stopping all running utils dockers"
docker rm -f $(docker ps | grep smurf-base | awk '{print $1}')
tmux rename-window -t ${tmux_session_name}:0 utils
tmux send-keys -t ${tmux_session_name}:0 'cd /home/cryo/docker/utils' C-m
tmux send-keys -t ${tmux_session_name}:0 './run.sh' C-m

if [ "$reboot" = true ] ; then

    # deactivate carriers
    echo "-> Deactivating LB and HB carriers"
    ssh root@shm-smrf-sp01 "clia deactivate board 4; clia deactivate board 5"
    
    echo "-> Waiting 5 sec before re-activating carriers"
    sleep 5
    
    # reactivate carriers
    echo "-> Re-activating LB and HB carriers"
    ssh root@shm-smrf-sp01 "clia activate board 4"
    ssh root@shm-smrf-sp01 "clia activate board 5"
    
    # wait for carrier ethernet to come up on carrier 4
    echo "-> Waiting for LB carrier ethernet to come up"
    cd $cpwd
    ./ping_carrier.sh 10.0.3.104
fi

################################################################################
### Configure slot4/LB

slot_number=4
start_slot_tmux ${slot_number}
pysmurf_docker_s4=`docker ps -n 1 -q`

if [ "$reboot" = true ] ; then
    config_pysmurf ${slot_number} ${pysmurf_docker_s4}
fi

### Done configuring slot4/LB 
################################################################################

# shut down slot 4 pyrogue server - right now can only run one pyrogue server at a time (at least when configuring)

if [ "$configure_hb" = true ] ; then
    # stop tailing smurf_server_s4 log
    tmux select-window -t smurf_slot4
    tmux select-pane -t 0
    tmux send-keys -t ${tmux_session_name}:4 C-c
    
    # stop smurf_server_s4
    stop_pyrogue 4
    
    ################################################################################
    ### Configure slot5/HB
    slot_number=5
    
    echo "-> Make sure HB carrier ethernet is up"
    cd $cpwd    
    ./ping_carrier.sh 10.0.3.105

    start_slot_tmux ${slot_number}
    pysmurf_docker_s5=`docker ps -n 1 -q`
    
    if [ "$reboot" = true ] ; then
	config_pysmurf ${slot_number} ${pysmurf_docker_s5}
    fi
    
    tmux attach -t ${tmux_session_name}
fi
    
