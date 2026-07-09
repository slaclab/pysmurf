#!/bin/bash

###############################################################################
# Terminal UI helpers
###############################################################################

# Detect color support
if [[ -t 1 ]]; then
    BOLD='\033[1m'
    DIM='\033[2m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[0;37m'
    BRED='\033[1;31m'
    BGREEN='\033[1;32m'
    BYELLOW='\033[1;33m'
    BBLUE='\033[1;34m'
    BCYAN='\033[1;36m'
    RESET='\033[0m'
else
    BOLD='' DIM='' RED='' GREEN='' YELLOW='' BLUE='' MAGENTA=''
    CYAN='' WHITE='' BRED='' BGREEN='' BYELLOW='' BBLUE='' BCYAN='' RESET=''
fi

header() {
    local text="$1"
    local width=60
    local pad=$(( (width - ${#text} - 2) / 2 ))
    local line=""
    for ((i=0; i<pad; i++)); do line+="═"; done
    printf "\n${BBLUE}%s %s %s${RESET}\n\n" "$line" "$text" "$line"
}

info() {
    printf "  ${CYAN}▸${RESET} %b\n" "$*"
}

success() {
    printf "  ${BGREEN}✓${RESET} %b\n" "$*"
}

warn() {
    printf "  ${BYELLOW}⚠${RESET}  %b\n" "$*"
}

error() {
    printf "  ${BRED}✗${RESET} %b\n" "$*"
}

step() {
    local current=$1 total=$2
    shift 2
    printf "  ${BOLD}[%d/%d]${RESET} %b\n" "$current" "$total" "$*"
}

dim() {
    printf "  ${DIM}%b${RESET}\n" "$*"
}

_SPINNER_PID=""
_SPINNER_START_TIME=""
spinner_start() {
    local msg="${1:-}"
    if [[ ! -t 1 ]]; then
        [[ -n "$msg" ]] && info "$msg"
        return
    fi
    _SPINNER_START_TIME=$(date +%s)
    (
        local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
        local i=0
        local start=$(date +%s)
        while true; do
            local elapsed=$(( $(date +%s) - start ))
            printf "\r\033[K  ${CYAN}%s${RESET} %s ${DIM}(%ds)${RESET}" "${frames[$i]}" "$msg" "$elapsed" >&2
            i=$(( (i + 1) % ${#frames[@]} ))
            sleep 0.1
        done
    ) &
    _SPINNER_PID=$!
    disown $_SPINNER_PID 2>/dev/null
}

spinner_stop() {
    local elapsed=""
    if [[ -n "$_SPINNER_PID" ]]; then
        kill $_SPINNER_PID 2>/dev/null
        wait $_SPINNER_PID 2>/dev/null
        printf "\r\033[K" >&2
        if [[ -n "$_SPINNER_START_TIME" ]]; then
            elapsed=$(( $(date +%s) - _SPINNER_START_TIME ))
        fi
        _SPINNER_PID=""
        _SPINNER_START_TIME=""
    fi
    echo "$elapsed"
}

spinner_stop_success() {
    local msg="$1"
    local elapsed=$(spinner_stop)
    if [[ -n "$elapsed" ]]; then
        success "${msg} ${DIM}(${elapsed}s)${RESET}"
    else
        success "${msg}"
    fi
}

# Status stage names for the parallel setup display
_stage_names=("off" "eth-up" "pyrogue" "docker" "server" "pysmurf" "setup" "done")
_stage_colors=("$RED" "$YELLOW" "$YELLOW" "$YELLOW" "$CYAN" "$CYAN" "$MAGENTA" "$GREEN")

slot_status_line() {
    local -n _slots_ref=$1
    local -n _status_ref=$2
    local line=""
    for ((i=0; i<${#_slots_ref[@]}; i++)); do
        local s=${_status_ref[$i]}
        local color=${_stage_colors[$s]}
        local label=${_stage_names[$s]}
        line+="$(printf "${BOLD}s%s${RESET}:${color}%-8s${RESET}" "${_slots_ref[$i]}" "$label")"
        if (( i < ${#_slots_ref[@]} - 1 )); then
            line+="  "
        fi
    done
    printf "\r\033[K  ${DIM}│${RESET} %b" "$line" >&2
}

###############################################################################
# Docker helpers
###############################################################################

matching_dockers () {
    if [ -z "$(docker ps | grep $1 | awk '{print $1}')" ] ; then
	return 0
    else
	return 1
    fi
}

stop_pyrogue () {
    info "Stopping pyrogue on slot ${BOLD}$1${RESET} ${DIM}($2)${RESET}";
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

wait_for_docker () {
    latest_docker=`docker ps -a -n 1 -q`
}

###############################################################################
# Slot setup helpers
###############################################################################

start_slot_tmux_and_pyrogue() {
    slot_number=$1
    pyrogue=$2

    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}

    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pyrogue} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh -N '${slot_number}'; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m
}

is_slot_pyrogue_up() {
    slot_number=$1
    if [[ -z `docker ps  | grep smurf_server_s${slot_number}`  ]]; then
	return 1
    fi
    return 0
}

is_slot_server_up() {
    slot_number=$1
    is_rogue_server_up ${slot_number}
    return $?
}

pysmurf_init() {
    slot_number=$1
    pysmurf_cfg=$2

    if [ -z "$pysmurf_cfg" ]
    then
	if [ ! -z ${pysmurf_init_script} ];
	then
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'ipython3 -i '${pysmurf_init_script}' '${slot_number} C-m
	else
	    error "Must either specify pysmurf_cfg for each slot in slot_cfgs,"
	    error "or specify a global pysmurf init script using the"
	    error "\`pysmurf_init_script\` variable in $startup_cfg."
	    error "Unable to start a pysmurf session."
	    exit 1
	fi
    else
	dim "Building pysmurf init script for slot ${slot_number}"
	tmp_pysmurf_init_script=/tmp/psmurf_init_`date +%s`.py
	zmq_port=$((9000 + 3*slot_number))

	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import pysmurf.client" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import matplotlib.pylab as plt" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import numpy as np" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import sys" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "sys.path = [path for path in sys.path if path != \".\"]" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import os" >> '${tmp_pysmurf_init_script} C-m

	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "server_port='${zmq_port}'" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "config_file=os.path.abspath(\"'${pysmurf_cfg}'\")" >> '${tmp_pysmurf_init_script} C-m

	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "S = pysmurf.client.SmurfControl(server_port=server_port,cfg_file=config_file,setup=False,make_logfile=False,data_dir=\"/data/\")" >> '${tmp_pysmurf_init_script} C-m

	tmux send-keys -t ${tmux_session_name}:${slot_number} "ipython3 -i ${tmp_pysmurf_init_script}" C-m
    fi
}

start_slot_pysmurf() {
    slot_number=$1
    pysmurf_cfg=$2

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

    spinner_start "Waiting for smurf_server_s${slot_number} docker to start"
    while [[ -z `docker ps  | grep smurf_server_s${slot_number}`  ]]; do
	sleep 1
    done
    spinner_stop_success "smurf_server_s${slot_number} docker started"

    spinner_start "Waiting for smurf_server_s${slot_number} rogue server (ZMQ heartbeat)"
    while ! is_rogue_server_up ${slot_number}
    do
	sleep 5
    done
    spinner_stop_success "smurf_server_s${slot_number} rogue server is up"

    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 1

    if [ "$enable_tmux_logging" = true ] ; then
	tmux run-shell -t ${tmux_session_name}:${slot_number} /home/cryo/tmux-logging/scripts/toggle_logging.sh
    fi
    pysmurf_init ${slot_number} ${pysmurf_cfg}

    latest_pysmurf_docker=`docker ps -a | grep pysmurf | grep -v pysmurf_s${slot_number} | head -n 1 | awk '{print $1}'`
}


run_pysmurf_setup () {
    slot_number=$1
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m
}

is_slot_pysmurf_setup_complete() {
    slot_number=$1
    tmux capture-pane -pt ${tmux_session_name}:${slot_number} -S -10 | grep -qE "Done with setup|Setup failed"
    ret=$?

    tmux capture-pane -pt ${tmux_session_name}:${slot_number} -S -10 | grep -Eom1 "Done with setup|Setup failed" | grep -q failed
    if [[ $? -eq 0 ]]; then
	error "Carrier in slot ${slot_number} failed to configure."
	error "Attach using \`tmux a -t ${tmux_session_name}\` to view errors."
	exit 1
    fi

    return $ret
}

config_pysmurf_serial () {
    slot_number=$1
    pysmurf_docker=$2

    tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m

    info "Running S.setup() on slot ${BOLD}${slot_number}${RESET} ${DIM}(docker: ${pysmurf_docker})${RESET}"

    sleep 2
    grep -qE "Done with setup|Setup failed" <(docker logs $pysmurf_docker -f)

    docker logs $pysmurf_docker | grep -Eom1 "Done with setup|Setup failed" | grep -q failed
    if [[ $? -eq 0 ]]; then
	error "Carrier in slot ${slot_number} failed to configure."
	error "Attach using \`tmux a -t ${tmux_session_name}\` to view errors."
	exit 1
    fi

    success "Slot ${slot_number} configured"

    if [ "$double_setup" = true ] ; then
	sleep 2
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.setup()' C-m
	sleep 5

	info "Running 2nd S.setup() on slot ${slot_number}"
	grep -q "Done with setup" <(docker logs $pysmurf_docker -f --since 0m)
	success "2nd setup complete on slot ${slot_number}"
    fi

    if [ "$disable_streaming" = true ] ; then
	info "Disabling streaming on slot ${slot_number}"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_stream_enable(0)' C-m
	sleep 2
    fi

    if [ "$write_config" = true ] ; then
	sleep 2
	info "Writing rogue config → /data/smurf_data/${ctime}_slot${slot_number}_config.yml"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_read_all(write_log=True); S.write_config("/data/smurf_data/'${ctime}'_slot'${slot_number}'_config.yml")' C-m
	until [ -s /data/smurf_data/${ctime}_slot${slot_number}_config.yml ]
	do
	    sleep 5
	done
	success "Config written for slot ${slot_number}"
    fi

    if [ "$save_state" = true ] ; then
	sleep 2
	info "Writing rogue state → /data/smurf_data/${ctime}_slot${slot_number}_state.yml"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'S.set_read_all(write_log=True); S.save_state("/data/smurf_data/'${ctime}'_slot'${slot_number}'_state.yml")' C-m
	until [ -s /data/smurf_data/${ctime}_slot${slot_number}_state.yml ]
	do
	    sleep 5
	done
	success "State written for slot ${slot_number}"
    fi

    if [ "$run_full_band_response" = true ] ; then
	sleep 2
	info "Running full band response on slot ${slot_number}"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'exec(open("scratch/shawn/full_band_response.py").read())' C-m
	grep -q "Done running full_band_response.py." <(docker logs $pysmurf_docker -f)
	success "Full band response complete on slot ${slot_number}"
    fi

    if [ "$run_half_band_test" = true ] ; then
	sleep 2
	info "Running half-band fill test on slot ${slot_number}"
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'sys.argv[1]='${ctime}'; exec(open("scratch/shawn/half_band_filling_test.py").read())' C-m
	grep -q "Done with half-band filling test." <(docker logs $pysmurf_docker -f)
	success "Half-band fill test complete on slot ${slot_number}"
    fi

    sleep 1
}

###############################################################################
# Heartbeat check (ZMQ)
###############################################################################

is_rogue_server_up(){
    slot=$1
    timeout_sec=10
    zmq_port=$((9000 + 3*slot))
    dockercmd="docker exec smurf_server_s${slot} python3 -c \"import pyrogue.interfaces; c = pyrogue.interfaces.SimpleClient('localhost', ${zmq_port}); print(c.getDisp('AMCc.LocalTime'))\""
    localtime=`eval ${dockercmd} 2>/dev/null`
    epochtime=`date "+%s" -d "$localtime" 2> /dev/null`

    if [ "$?" -eq "0" ]; then
	val0=$epochtime
	val1=$epochtime
	ctime0=`date +%s`

	dim "Got timestamp from smurf_server_s${slot}, verifying increment..."
	while [[ $val0 -eq $val1 ]]; do
	    localtime1=`eval ${dockercmd} 2>/dev/null`
	    epochtime1=`date "+%s" -d "$localtime1"`
	    if [  "$?" -eq "0" ]; then
		val1=$epochtime1
	    fi

	    ctime1=`date +%s`
	    if [ "$(($ctime1-$ctime0))" -gt $timeout_sec ]; then
		warn "Timed out waiting for smurf_server_s${slot} timestamp to increment (${timeout_sec}s)"
		return 1
	    fi
	done
    else
	return 1
    fi

    success "smurf_server_s${slot} timestamp incrementing — server is up"
    return 0
}
