#!/bin/bash

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
    # stop pyrogue dockers
    echo "-> Stopping slot $1 pyrogue docker $2";
    cd $2;
    ./stop.sh -N $1;
}

wait_for_docker_instance () {
    keyword=$1
    cadence_sec=1
    while [[ -z $(docker ps | grep $keyword | awk '{print $1}') ]];
    do
	sleep ${cadence_sec}
    done
    dockerid=$(docker ps | grep $keyword | awk '{print $1}' | head -n 1)
    echo $dockerid
}

# I hate this
wait_for_docker () {
    latest_docker=`docker ps -a -n 1 -q`    
}

start_slot_tmux_and_pyrogue() {
    slot_number=$1
    pyrogue=$2

    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}

    # what is this??
    #tmux send-keys -l -t ${tmux_session_name}:${slot_number} C-b S-p
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pyrogue} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh -N '${slot_number}'; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m
}

is_slot_pyrogue_up() {
    slot_number=$1
    if [[ -z `docker ps  | grep smurf_server_s${slot_number}`  ]]; then
	#echo '-> smurf_server_s'${slot_number}' docker started.'
	return 1
    fi
    return 0
}

is_slot_server_up() {
    slot_number=$1

    ## Old, bad way of determining if server is up
    #tmux capture-pane -pt ${tmux_session_name}:${slot_number} | grep -q "Running GUI."
    #return $?
    
    is_rogue_server_up ${slot_number}
    return $?
}

pysmurf_init() {
    slot_number=$1
    pysmurf_cfg=$2

    if [ -z "$pysmurf_cfg" ]
    then
	# used to have an institution init script, specified in the
	# smurf_startup.cfg script as `pysmurf_init_script`
	if [ ! -z ${pysmurf_init_script} ];
	then
	    # using old init script pysmurf startup
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'ipython3 -i '${pysmurf_init_script}' '${slot_number} C-m
	else
	    echo "Must either specify pysmurf_cfg for each slot in slot_cfgs," 1>&2
	    echo "or specify a global pysmurf init script using the" 1>&2
	    echo '`pysmurf_init_script`'" variable in $startup_cfg." 1>&2
	    echo "Unable to start a pysmurf session." 1>&2
	    exit 1	    
	fi
    else
	tmp_pysmurf_init_script=/tmp/psmurf_init_`date +%s`.py

	# load pysmurf.client
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import pysmurf.client" >> '${tmp_pysmurf_init_script} C-m

	# load 3rd party libraries
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import matplotlib.pylab as plt" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import numpy as np" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import sys" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import os" >> '${tmp_pysmurf_init_script} C-m	

	# define some local variables that some shawnhammer functions will use later
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "epics_prefix=\"smurf_server_s'${slot_number}'\"" >> '${tmp_pysmurf_init_script} C-m
	# abs path in case user specified the relative path
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "config_file=os.path.abspath(\"'${pysmurf_cfg}'\")" >> '${tmp_pysmurf_init_script} C-m

	# instantiate pysmurf
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "S = pysmurf.client.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False,shelf_manager=\"'${shelfmanager}'\")" >> '${tmp_pysmurf_init_script} C-m

	# start in new ipython session
	tmux send-keys -t ${tmux_session_name}:${slot_number} "ipython3 -i ${tmp_pysmurf_init_script}" C-m	
    fi    
}

start_slot_pysmurf() {
    slot_number=$1
    pysmurf_cfg=$2
    
    # start pysmurf in a split window and initialize the carrier
    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 1

    if [ "$enable_tmux_logging" = true ] ; then
	tmux run-shell -t ${tmux_session_name}:${slot_number} /home/cryo/tmux-logging/scripts/toggle_logging.sh
    fi    
    pysmurf_init ${slot_number} ${pysmurf_cfg}
}

start_slot_tmux_serial () {
    slot_number=$1
    pyrogue=$2
    pysmurf_cfg=$3

    pysmurf_docker0=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
    
    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}
    
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '$2 C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh -N '${slot_number}'; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m

    echo '-> Waiting for smurf_server_s'${slot_number}' docker to start.'
    while [[ -z `docker ps  | grep smurf_server_s${slot_number}`  ]]; do
	sleep 1
    done
    echo '-> smurf_server_s'${slot_number}' docker started.'    

    # Old, terrible way, relies on parsing text in the rogue server docker log
    #echo '-> Waiting for smurf_server_s'${slot_number}' GUI to come up.'    
    #sleep 2
    #grep -q "Running GUI." <(docker logs smurf_server_s${slot_number} -f)

    echo '-> Waiting for smurf_server_s'${slot_number}' to come up ...'        
    # Checks for incremeting LocalTime
    while ! is_rogue_server_up ${slot_number}
    do
	sleep 5
    done
    
    # start pysmurf in a split window and initialize the carrier
    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 1

    if [ "$enable_tmux_logging" = true ] ; then
	tmux run-shell -t ${tmux_session_name}:${slot_number} /home/cryo/tmux-logging/scripts/toggle_logging.sh
    fi
    #tmux send-keys -t ${tmux_session_name}:${slot_number} 'ipython3 -i '${pysmurf_init_script}' '${slot_number} C-m
    pysmurf_init ${slot_number} ${pysmurf_cfg}    

    ## not the safest way to do this.  If someone else starts a
    ## pysmurf docker, will satisfy this condition.  Not even sure why
    ## this is even needed - why is there so much latency on
    ## smurf-srv03 between starting a docker and it showing up in
    ## docker ps?
    echo "pysmurf_docker0=$pysmurf_docker0"
    latest_pysmurf_docker=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
    echo "latest_pysmurf_docker=$latest_pysmurf_docker"	    
#    while [ "$pysmurf_docker0" == "$latest_pysmurf_docker" ]; do
#    	latest_pysmurf_docker=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
#    	sleep 1
#	echo "latest_pysmurf_docker=$latest_pysmurf_docker"		
#    done
    
    # after running this, can run
    # pysmurf_docker=`docker ps -n 1 -q`
    # to hex of most recently created docker.
}


run_pysmurf_setup () {
    slot_number=$1
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m
}

is_slot_pysmurf_setup_complete() {
    slot_number=$1
    tmux capture-pane -pt ${tmux_session_name}:${slot_number} -S -10 | grep -qE "Done with setup|Setup failed"
    ret=$?
    
    # check if setup failed
    tmux capture-pane -pt ${tmux_session_name}:${slot_number} -S -10 | grep -Eom1 "Done with setup|Setup failed" | grep -q failed
    if [[ $? -eq 0 ]]; then
	echo '-> Carrier failed to configure.  Attach using `tmux a` to view errors.'
	exit 1
    fi    
    
    return $ret
}

# right now, real dumb.  Assumes the active window in tmux is this
# slot's
config_pysmurf_serial () {
    slot_number=$1
    pysmurf_docker=$2
    
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m    
    
    # wait for setup to complete
    echo "-> Waiting for carrier setup (watching pysmurf docker ${pysmurf_docker})"

    # not clear why, but on smurf-srv03 need this wait or attempt to
    # wait until done with setup fails.
    sleep 2
    grep -qE "Done with setup|Setup failed" <(docker logs $pysmurf_docker -f)

    # check if setup failed
    docker logs $pysmurf_docker | grep -Eom1 "Done with setup|Setup failed" | grep -q failed
    if [[ $? -eq 0 ]]; then
	echo '-> Carrier failed to configure.  Attach using `tmux a` to view errors.'
	exit 1
    fi
    
    echo "-> Carrier is configured."
    
    if [ "$double_setup" = true ] ; then
	sleep 2    
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m    
	sleep 5
	
	# wait for setup to complete
	echo "-> Waiting for 2nd carrier setup (watching pysmurf docker ${pysmurf_docker})"
	# since second one, need the --since flag or else will catch
	# the first
	grep -q "Done with setup" <(docker logs $pysmurf_docker -f --since 0m)
	echo "-> Done with 2nd setup"	
    fi

    if [ "$disable_streaming" = true ] ; then    
	echo "-> Disable streaming (unless taking data)"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_stream_enable(0)' C-m
	sleep 2
    fi

    # actually, write state
    if [ "$write_config" = true ] ; then
	sleep 2
	echo "-> Writing rogue configuration to /data/smurf_data/${ctime}_slot${slot_number}_config.yml"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_read_all(write_log=True); S.write_config("/data/smurf_data/'${ctime}'_slot'${slot_number}'_config.yml")' C-m
	# wait until file exists
	echo "-> Waiting for /data/smurf_data/${ctime}_slot${slot_number}_config.yml to be written to disk ..."
	until [ -s /data/smurf_data/${ctime}_slot${slot_number}_config.yml ]
	do
	    sleep 5
	done
    fi

    # actually, write state
    if [ "$save_state" = true ] ; then
	sleep 2
	echo "-> Writing rogue state to /data/smurf_data/${ctime}_slot${slot_number}_state.yml"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_read_all(write_log=True); S.save_state("/data/smurf_data/'${ctime}'_slot'${slot_number}'_state.yml")' C-m
	# wait until file exists
	echo "-> Waiting for /data/smurf_data/${ctime}_slot${slot_number}_state.yml to be written to disk ..."
	until [ -s /data/smurf_data/${ctime}_slot${slot_number}_state.yml ]
	do
	    sleep 5
	done
    fi    

    if [ "$run_full_band_response" = true ] ; then    
	sleep 2
	echo "-> Running full band response across all configured bands."
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'exec(open("scratch/shawn/full_band_response.py").read())' C-m    
	grep -q "Done running full_band_response.py." <(docker logs $pysmurf_docker -f)
    fi
    
    if [ "$run_half_band_test" = true ] ; then    
	sleep 2
	echo "-> Running half-band fill test"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'sys.argv[1]='${ctime}'; exec(open("scratch/shawn/half_band_filling_test.py").read())' C-m    
	grep -q "Done with half-band filling test." <(docker logs $pysmurf_docker -f)
    fi
    
    sleep 1
}

is_rogue_server_up(){
    slot=$1
    timeout_sec=10
    dockercmd="docker exec -it smurf_utils caget -w 1.0 -t smurf_server_s${slot}:AMCc:LocalTime -S"
    localtime=`${dockercmd}`
    epochtime=`date "+%s" -d "$localtime" 2> /dev/null`
    
    if [ "$?" -eq "0" ]; then
	# read back a valid timestamp - server must be up.
	val0=$epochtime
	val1=$epochtime
	# when we got the first valid timestamp back from the server
	ctime0=`date +%s`
	
	echo "-> Got a valid timestamp back from smurf_server_s${slot}, checking to confirm it's incrementing ..."
	while [[ $val0 -eq $val1 ]]; do
	    localtime1=`${dockercmd}`
	    epochtime1=`date "+%s" -d "$localtime1"`
	    if [  "$?" -eq "0" ]; then
		val1=$epochtime1
	    fi
	    
	    # time out if timestamp from server is not incrementing
	    ctime1=`date +%s`
	    if [ "$(($ctime1-$ctime0))" -gt $timeout_sec ]; then
		echo "-> Timed out waiting for smurf_server_s${slot} timestamp to increment (waited $timeout_sec seconds)!"
		return 1
	    fi
	done
    else
	return 1
    fi

    # success!
    echo "-> smurf_server_s${slot} timestamp is incrementing, server is up."    
    return 0
}

