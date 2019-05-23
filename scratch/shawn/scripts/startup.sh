#!/bin/bash

reboot=true

master_tmux_name=smurf

stop_pyrogue () {
    # stop pyrogue dockers, assumes the pyrogue version you want to use
    # has been soft linked to slotN/current
    echo "-> Stopping slot $1 pyrogue docker";
    cd /home/cryo/docker/smurf/slot$1/current;
    ./stop.sh;
}

# use tmux instead
# https://www.peterdebelak.com/blog/tmux-scripting/

# kill smurf tmux, if running, and restart
echo "-> Killing ${master_tmux_name} tmux session"
tmux kill-session -t ${master_tmux_name}
echo "-> Starting a new ${master_tmux_name} tmux session"
tmux new -s ${master_tmux_name} -d

# stop pyrogue servers on both carriers
stop_pyrogue 4
stop_pyrogue 5

# stop all pysmurf dockers
echo "-> Stopping all running pysmurf dockers"
docker rm -f $(docker ps | grep pysmurf | awk '{print $1}')

## will need a utils docker.  first remove all others to avoid
## proliferation, then start one in tmux
echo "-> Stopping all running utils dockers"
docker rm -f $(docker ps | grep smurf-base | awk '{print $1}')
tmux rename-window -t ${master_tmux_name} utils
tmux send-keys -t ${master_tmux_name} 'cd /home/cryo/docker/utils' C-m
tmux send-keys -t ${master_tmux_name} './run.sh' C-m

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
    /home/cryo/shawn/ping_carrier.sh 10.0.3.104
fi

################################################################################
### Configure slot4/LB 

tmux new-window -t ${master_tmux_name}
tmux rename-window -t ${master_tmux_name} smurf_slot4
tmux send-keys -t ${master_tmux_name} 'cd /home/cryo/docker/smurf/slot4/current' C-m
tmux send-keys -t ${master_tmux_name} './run.sh; sleep 5; docker logs smurf_server_s4 -f' C-m

# might be nicer to wait for heartbeat, like
#tmux send-keys -t utils 'caget -w 60 smurf_server_s4:AMCc:Time; tmux wait-for -S smurf-server-s4-up' C-m\; wait-for smurf-server-s4-up
# but would need tmux installed in utils
# short sleep to make sure smurf_server_s4 is running
sleep 2
grep -q "Starting GUI" <(docker logs smurf_server_s4 -f)

# start pysmurf in a split window and initialize the carrier
tmux split-window -v -t ${master_tmux_name}
tmux send-keys -t ${master_tmux_name} 'cd /home/cryo/docker/pysmurf/dspv3' C-m
tmux send-keys -t ${master_tmux_name} './run.sh' C-m
sleep 1
pysmurf_s4_docker=`docker ps -n 1 -q`
# needed a way to see when setup completes.  For now, tee pysmurf commands and output to a file on the main
# file system for parsing.
#rm -v /data/pysmurf_tmux_sessions/pysmurf_s4.log
#tmux send-keys -t ${master_tmux_name} 'ipython3 -i scratch/eyy/ipython_start/init_dspv3.py 4 | tee /data/pysmurf_tmux_sessions/pysmurf_s4.log' C-m
tmux send-keys -t ${master_tmux_name} 'ipython3 -i scratch/eyy/ipython_start/init_dspv3.py 4' C-m

if [ "$reboot" = true ] ; then
    tmux send-keys -t ${master_tmux_name} 'S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=True,make_logfile=False)' C-m

    # wait for setup to complete
    echo "-> Waiting for carrier setup"
    grep -q "Done with setup" <(docker logs $pysmurf_s4_docker -f)
    echo "-> Carrier is configured"

    echo "-> Disable streaming (unless taking data)"
    tmux send-keys -t ${master_tmux_name} 'S.set_stream_enable(0)' C-m
    sleep 1
fi

### Done configuring slot4/LB 
################################################################################

# shut down slot 4 pyrogue server - right now can only run one pyrogue server at a time (at least when configuring)
stop_pyrogue 4

tmux attach -t ${master_tmux_name}
    
