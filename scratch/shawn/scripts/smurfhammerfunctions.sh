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

_spin_wait_elapsed=0

# Synchronous spinner — no background processes.
# Usage: spin_wait "message" "condition_command"
# Loops until condition_command succeeds (exit 0), showing a braille spinner.
# Sets _spin_wait_elapsed to the number of seconds waited.
spin_wait() {
    local msg="$1"
    local condition="$2"
    local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local i=0
    local start=$(date +%s)
    _spin_wait_elapsed=0

    if [[ ! -t 1 ]]; then
        info "$msg"
        while ! eval "$condition"; do sleep 1; done
        _spin_wait_elapsed=$(( $(date +%s) - start ))
        return
    fi

    printf "\033[?25l"
    trap 'printf "\033[?25h\n"; exit 130' INT TERM

    while ! eval "$condition"; do
        _spin_wait_elapsed=$(( $(date +%s) - start ))
        printf "\r\033[K  ${CYAN}%s${RESET} %s ${DIM}(%ds)${RESET}" "${frames[$i]}" "$msg" "$_spin_wait_elapsed"
        i=$(( (i + 1) % ${#frames[@]} ))
        sleep 0.2
    done

    _spin_wait_elapsed=$(( $(date +%s) - start ))
    printf "\r\033[K\033[?25h"
}


###############################################################################
# Rogue version detection
###############################################################################

# Cache of rogue major version per slot (avoids repeated docker exec)
declare -A _rogue_major_version

get_rogue_major_version() {
    local slot=$1
    if [[ -z "${_rogue_major_version[$slot]:-}" ]]; then
        local ver
        ver=$(docker exec smurf_server_s${slot} python3 -c "import pyrogue; print(pyrogue.__version__)" 2>/dev/null | head -1)
        _rogue_major_version[$slot]="${ver%%.*}"
    fi
    echo "${_rogue_major_version[$slot]}"
}

is_rogue6() {
    local slot=$1
    local major=$(get_rogue_major_version $slot)
    [[ "$major" -ge 6 ]] 2>/dev/null
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

# Full setup sequence for one slot. Runs in a subshell (backgrounded).
# Produces the same rich output as the old serial path.
setup_slot() {
    local slot_number=$1
    local pyrogue=$2
    local pysmurf_cfg=$3
    local carrier_ip="10.0.${crate_id}.$((slot_number + 100))"

    # Wait for carrier ethernet
    spin_wait "Slot ${slot_number}: Waiting for ${carrier_ip}" \
        "timeout 0.5 ping -c 1 -n ${carrier_ip} &>/dev/null"
    success "Slot ${slot_number}: ${carrier_ip} is online ${DIM}(${_spin_wait_elapsed}s)${RESET}"

    # Start server docker
    check_docker_pull ${slot_number} "${pyrogue}"
    start_slot_tmux_and_pyrogue ${slot_number} ${pyrogue}

    spin_wait "Slot ${slot_number}: Waiting for smurf_server_s${slot_number} docker" \
        "is_slot_pyrogue_up ${slot_number}"
    success "Slot ${slot_number}: smurf_server_s${slot_number} docker started ${DIM}(${_spin_wait_elapsed}s)${RESET}"

    # Monitor server startup (FW mismatch, PROM, setDefaults, heartbeat)
    monitor_server_startup ${slot_number}
    if [[ $? -ne 0 ]]; then
        error "Slot ${slot_number}: Server startup failed"
        return 1
    fi

    # Start pysmurf client
    start_slot_pysmurf ${slot_number} "${pysmurf_cfg}"

    # Run S.setup() if configured
    if [ "${configure_pysmurf}" = true ]; then
        run_pysmurf_setup ${slot_number}
        info "Slot ${slot_number}: Running S.setup()"

        # Wait for setup to complete
        local setup_start=$(date +%s)
        while true; do
            if is_slot_pysmurf_setup_complete ${slot_number}; then
                success "Slot ${slot_number}: S.setup() complete ${DIM}($(( $(date +%s) - setup_start ))s)${RESET}"
                break
            fi
            sleep 2
        done
    fi

    success "Slot ${slot_number}: Done"
}

start_slot_tmux_and_pyrogue() {
    slot_number=$1
    pyrogue=$2

    tmux new-window -t ${tmux_session_name}:${slot_number}
    tmux rename-window -t ${tmux_session_name}:${slot_number} smurf_slot${slot_number}

    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pyrogue} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} \
        '. .env; extra=""; [ -c /dev/datadev_0 ] && [ -c /dev/datadev_1 ] && extra="-f docker-compose.pcie.yml"; slot='${slot_number}' docker-compose -f docker-compose.yml $extra up -d smurf_server_s'${slot_number}'; sleep 5; docker logs smurf_server_s'${slot_number}' -f' C-m
}

is_slot_pyrogue_up() {
    slot_number=$1
    docker ps --filter "name=^smurf_server_s${slot_number}$" --format '{{.ID}}' 2>/dev/null | grep -q .
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

	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import pysmurf.client" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import matplotlib.pylab as plt" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import numpy as np" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import sys" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "sys.path = [path for path in sys.path if path != \".\"]" >> '${tmp_pysmurf_init_script} C-m
	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "import os" >> '${tmp_pysmurf_init_script} C-m

	tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "config_file=os.path.abspath(\"'${pysmurf_cfg}'\")" >> '${tmp_pysmurf_init_script} C-m

	if is_rogue6 ${slot_number}; then
	    # Rogue 6+: connect via ZMQ server_port
	    zmq_port=$((9000 + 3*slot_number))
	    dim "Rogue 6 detected on slot ${slot_number} (ZMQ port ${zmq_port})"
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "server_port='${zmq_port}'" >> '${tmp_pysmurf_init_script} C-m
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "S = pysmurf.client.SmurfControl(server_port=server_port,cfg_file=config_file,setup=False,make_logfile=False,data_dir=\"/data/\")" >> '${tmp_pysmurf_init_script} C-m
	else
	    # Rogue 4: connect via EPICS epics_root
	    dim "Rogue 4 detected on slot ${slot_number} (EPICS)"
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "epics_prefix=\"smurf_server_s'${slot_number}'\"" >> '${tmp_pysmurf_init_script} C-m
	    tmux send-keys -t ${tmux_session_name}:${slot_number} 'echo "S = pysmurf.client.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False,shelf_manager=\"'${shelfmanager}'\")" >> '${tmp_pysmurf_init_script} C-m
	fi

	tmux send-keys -t ${tmux_session_name}:${slot_number} "ipython3 -i ${tmp_pysmurf_init_script}" C-m
    fi
}

start_slot_pysmurf() {
    slot_number=$1
    pysmurf_cfg=$2

    docker rm -f "pysmurf_s${slot_number}" 2>/dev/null

    tmux split-window -v -t ${tmux_session_name}:${slot_number}
    tmux send-keys -t ${tmux_session_name}:${slot_number} 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:${slot_number} './run.sh' C-m
    sleep 2

    local new_pysmurf_id
    new_pysmurf_id=$(docker ps -a -n 1 -q)
    if [[ -n "$new_pysmurf_id" ]]; then
        docker rename "$new_pysmurf_id" "pysmurf_s${slot_number}" 2>/dev/null
    fi

    if [ "$enable_tmux_logging" = true ] ; then
	tmux run-shell -t ${tmux_session_name}:${slot_number} /home/cryo/tmux-logging/scripts/toggle_logging.sh
    fi
    pysmurf_init ${slot_number} ${pysmurf_cfg}
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


###############################################################################
# Server startup log monitoring
###############################################################################

# Monitors the server docker log during startup, reporting:
# - Docker image pull progress
# - FPGA firmware mismatch & reprogramming (erase/write/verify PROM)
# - setDefaults retry failures
# - Heartbeat detection
# After completion, prints a success message.
#
# Usage: monitor_server_startup <slot_number>
monitor_server_startup() {
    local slot_number=$1
    local container="smurf_server_s${slot_number}"
    local saw_fw_mismatch=false
    local saw_program=false
    local saw_setdefaults_error=false
    local setdefaults_try=0
    local heartbeat_ok=false
    local setdefaults_started=false
    local setdefaults_done=false

    local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local fi=0
    local start=$(date +%s)
    local status_msg="Waiting for rogue server to start"

    if [[ -t 1 ]]; then
        printf "\033[?25l"
    fi

    while true; do
        local elapsed=$(( $(date +%s) - start ))

        # Poll the latest log content (no buffering issues)
        local logs
        logs=$(docker logs "$container" 2>&1)

        # Check for firmware mismatch
        if ! $saw_fw_mismatch && echo "$logs" | grep -q "They don't match"; then
            saw_fw_mismatch=true
            local old_hash=$(echo "$logs" | grep "Firmware githash:" | grep -oP "'[a-f0-9]+'" | tr -d "'")
            local new_hash=$(echo "$logs" | grep "MCS file githash:" | grep -oP "'[a-f0-9]+'" | tr -d "'")
            if [[ -t 1 ]]; then printf "\r\033[K"; fi
            warn "FPGA firmware mismatch — reprogramming PROM"
            dim "  FPGA: ${old_hash:-?}  MCS: ${new_hash:-?}"
            status_msg="Erasing PROM"
        fi

        # Track PROM programming progress
        if $saw_fw_mismatch && ! $saw_program; then
            local latest_erase=$(echo "$logs" | grep -o "Erasing the PROM: [0-9]* percent" | tail -1)
            local latest_write=$(echo "$logs" | grep -o "Writing the PROM: [0-9]* percent" | tail -1)
            local latest_verify=$(echo "$logs" | grep -o "Verifying the PROM: [0-9]* percent" | tail -1)

            if [[ -n "$latest_verify" ]]; then
                local pct=$(echo "$latest_verify" | grep -oP '[0-9]+')
                status_msg="Verifying PROM (${pct}%)"
            elif [[ -n "$latest_write" ]]; then
                local pct=$(echo "$latest_write" | grep -oP '[0-9]+')
                status_msg="Writing PROM (${pct}%)"
            elif [[ -n "$latest_erase" ]]; then
                local pct=$(echo "$latest_erase" | grep -oP '[0-9]+')
                status_msg="Erasing PROM (${pct}%)"
            fi

            if echo "$logs" | grep -q "FPGA programmed successfully"; then
                saw_program=true
                if [[ -t 1 ]]; then printf "\r\033[K"; fi
                success "FPGA programmed successfully"
                status_msg="Waiting for FPGA reboot"
            fi
        fi

        # Track FPGA reboot / ETH
        if $saw_program && ! $heartbeat_ok; then
            if echo "$logs" | grep -q "FPGA's ETH came up"; then
                status_msg="FPGA rebooted, starting server"
            fi
        fi

        # Detect heartbeat (server ZMQ/EPICS is responding)
        if ! $heartbeat_ok && echo "$logs" | grep -q "Running. Hit cntrl-c"; then
            if is_rogue_server_up ${slot_number} >/dev/null 2>&1; then
                heartbeat_ok=true
                if [[ -t 1 ]]; then printf "\r\033[K"; fi
                success "smurf_server_s${slot_number} rogue heartbeat detected ${DIM}(${elapsed}s)${RESET}"
                status_msg="Server running setDefaults"
            fi
        fi

        # Track setDefaults progress (runs after heartbeat is up)
        if echo "$logs" | grep -q "^Setting defaults from file"; then
            setdefaults_started=true
            local current_try=$(echo "$logs" | grep -c "^Setting defaults from file")
            local retry_cnt=$(echo "$logs" | grep -o "retryCnt = [0-9]*" | tail -1 | grep -oP '[0-9]+')
            if [[ -n "$retry_cnt" && "$retry_cnt" -gt 0 ]]; then
                status_msg="setDefaults try ${current_try}, AppTop.Init() retry ${retry_cnt}/7"
            else
                status_msg="setDefaults try ${current_try}, running AppTop.Init()"
            fi
        fi

        # Check for individual setDefaults try failure
        if echo "$logs" | grep -q "ERROR: Setting defaults try number\|Setting defaults try number .* failed"; then
            local fail_try=$(echo "$logs" | grep -oP "(?:ERROR: Setting defaults try number|Setting defaults try number )\K[0-9]+" | tail -1)
            if [[ -n "$fail_try" && "$fail_try" != "$setdefaults_try" ]]; then
                setdefaults_try="$fail_try"
                if [[ -t 1 ]]; then printf "\r\033[K"; fi
                warn "setDefaults try ${fail_try} failed (JESD init retries exhausted)"
                saw_setdefaults_error=true
            fi
        fi

        # Check for "LoadConfig process did not finish" (timeout-based failure)
        if echo "$logs" | grep -q 'process did not finish'; then
            local timeout_try=$(echo "$logs" | grep -c "process did not finish")
            if [[ "$timeout_try" -gt "${_last_timeout_warn:-0}" ]]; then
                _last_timeout_warn=$timeout_try
                if [[ -t 1 ]]; then printf "\r\033[K"; fi
                warn "setDefaults LoadConfig timed out (try ${timeout_try})"
                saw_setdefaults_error=true
            fi
        fi

        # Check for fatal failure (all retries exhausted)
        if echo "$logs" | grep -q "Failed to set defaults after"; then
            if [[ -t 1 ]]; then printf "\r\033[K\033[?25h"; fi
            error "Server gave up on setDefaults — JESD link won't lock"
            dim "  View log: docker logs ${container} | tail -80"
            return 1
        fi

        # Check for setDefaults success
        if echo "$logs" | grep -q "Defaults were set correctly"; then
            setdefaults_done=true
            if [[ -t 1 ]]; then printf "\r\033[K"; fi
            success "setDefaults completed"
            break
        fi

        # If heartbeat is up but no setDefaults after 15s, server likely
        # started without configure=True — don't wait forever
        if $heartbeat_ok && ! $setdefaults_started; then
            local since_heartbeat=$(( elapsed - ${_heartbeat_time:-$elapsed} ))
            if [[ -z "${_heartbeat_time:-}" ]]; then
                _heartbeat_time=$elapsed
            elif [[ $since_heartbeat -gt 15 ]]; then
                setdefaults_done=true
                break
            fi
        fi

        # Render spinner
        if [[ -t 1 ]]; then
            printf "\r\033[K  ${CYAN}%s${RESET} %s ${DIM}(%ds)${RESET}" \
                "${frames[$fi]}" "$status_msg" "$elapsed"
            fi=$(( (fi + 1) % ${#frames[@]} ))
        fi

        sleep 1
    done

    local total_elapsed=$(( $(date +%s) - start ))
    if [[ -t 1 ]]; then printf "\r\033[K\033[?25h"; fi

    if ! $heartbeat_ok; then
        success "smurf_server_s${slot_number} rogue server is up ${DIM}(${total_elapsed}s)${RESET}"
    fi

    if $saw_fw_mismatch; then
        dim "  Note: FPGA was reprogrammed during this startup"
    fi
    if $saw_setdefaults_error; then
        dim "  Note: setDefaults required multiple attempts"
    fi
    dim "  Tip: docker logs ${container} -f"
    return 0
}

# Monitor a docker-compose pull/start and detect if images are being pulled.
# Call this before waiting for the container to appear.
# Usage: check_docker_pull <slot_number> <pyrogue_dir>
check_docker_pull() {
    local slot_number=$1
    local pyrogue_dir=$2
    local container="smurf_server_s${slot_number}"

    # If docker is already running, no pull needed
    if docker ps --filter "name=^${container}$" --format '{{.ID}}' 2>/dev/null | grep -q .; then
        return 0
    fi

    # Check if image exists locally
    local image
    image=$(grep -r "image:" "$pyrogue_dir/docker-compose.yml" 2>/dev/null | head -1 | awk '{print $2}')
    if [[ -n "$image" ]] && ! docker image inspect "$image" >/dev/null 2>&1; then
        info "Docker image ${BOLD}${image}${RESET} not cached — will be pulled"
        dim "  This may take a few minutes on first run"
    fi
}

###############################################################################
# Heartbeat check
###############################################################################

is_rogue_server_up(){
    slot=$1
    timeout_sec=10

    if is_rogue6 ${slot}; then
	# Rogue 6+: query LocalTime via ZMQ
	zmq_port=$((9000 + 3*slot))
	dockercmd="docker exec smurf_server_s${slot} python3 -c \"import pyrogue.interfaces; c = pyrogue.interfaces.SimpleClient('localhost', ${zmq_port}); print(c.getDisp('AMCc.LocalTime'))\" 2>/dev/null | head -1"
    else
	# Rogue 4: query LocalTime via EPICS caget
	dockercmd="docker exec smurf_server_s${slot} caget -w 1.0 -t smurf_server_s${slot}:AMCc:LocalTime -S 2>/dev/null"
    fi

    localtime=`eval ${dockercmd}`
    epochtime=`date "+%s" -d "$localtime" 2> /dev/null`

    if [ "$?" -eq "0" ]; then
	val0=$epochtime
	val1=$epochtime
	ctime0=`date +%s`

	while [[ $val0 -eq $val1 ]]; do
	    localtime1=`eval ${dockercmd}`
	    epochtime1=`date "+%s" -d "$localtime1" 2> /dev/null`
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

    success "smurf_server_s${slot} timestamp incrementing - server is up"
    return 0
}
