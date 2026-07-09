#!/bin/bash

#default startup cfg
startup_cfg=/data/smurf_startup_cfg/smurf_startup.cfg

usage() {
    cat <<'EOF'

  ┌─────────────────────────────────────────────────────────────┐
  │                      SMURFHAMMER                            │
  │         SMuRF system startup & configuration tool           │
  └─────────────────────────────────────────────────────────────┘

  Usage: smurfhammer [OPTIONS]

  Options:
    -c <path>    Path to startup config file
                 (default: /data/smurf_startup_cfg/smurf_startup.cfg)
    -i <script>  pysmurf init script (relative to $pysmurf/pysmurf/)
    -t           Run thermal test after setup
    -h           Show this help message

  Description:
    Brings up SMuRF readout electronics by orchestrating pyrogue server
    dockers, pysmurf client sessions, and carrier configuration across
    one or more ATCA slots or RFSoC targets.

    The startup config file defines which slots to configure, which
    pyrogue docker to assign to each, and various options (reboot,
    parallel setup, timing master, etc).

  Workflow:
    1. Load startup configuration
    2. Kill existing tmux session and dockers
    3. (Optional) Reboot carriers via shelf manager
    4. Start pyrogue servers and wait for ZMQ heartbeat
    5. Launch pysmurf client sessions
    6. (Optional) Run S.setup() on each carrier
    7. (Optional) Run post-setup scripts

  Prerequisites:
    - SSH key auth must be configured for the shelf manager
      (ssh-copy-id root@<shelfmanager>) or carrier reboot/fan
      control commands will prompt for a password and hang.

  Examples:
    smurfhammer
    smurfhammer -c /data/smurf_startup_cfg/my_custom.cfg
    smurfhammer -c /data/smurf_startup_cfg/my_custom.cfg -t

EOF
    exit 0
}

while getopts ":hic:t" opt; do
    case ${opt} in
      h )
          usage
          ;;
      i )
	  # make sure this file exists
	  if [ ! -f $pysmurf/pysmurf/$OPTARG ]; then
              echo "Invalid Input: No file found at $pysmurf/pysmurf/$OPTARG" 1>&2
              exit 1
	  fi
	  pysmurf_init_script=$OPTARG
          ;;
      c )
	  # make sure this file exists
	  if [ ! -f $OPTARG ]; then
              echo "Invalid Input: No file found at $OPTARG" 1>&2
              exit 1
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
    echo "$startup_cfg doesn't exist, unable to smurfhammer." 1>&2
    exit 1
fi
source ${startup_cfg}

# source functions (and UI helpers)
source smurfhammerfunctions

# Record start time for elapsed calculation
_hammer_start=`date +%s`
ctime=${_hammer_start}

###############################################################################
# Banner
###############################################################################

printf "\n"
printf "  ${BOLD}░▒▓ SMURFHAMMER ▓▒░${RESET}\n"
printf "  ${DIM}SMuRF system startup - $(date '+%Y-%m-%d %H:%M:%S')${RESET}\n"
printf "\n"
printf "  ${DIM}Config:${RESET}  %s\n" "$startup_cfg"

###############################################################################
# Load configurations
###############################################################################

header "CONFIGURATION"

# if enable-tmux-logging, check that the tmux-logging plugin is installed
if [[ "$enable_tmux_logging" = true && ! -d "/home/cryo/tmux-logging" ]] ; then
    warn "tmux logging enabled but plugin not found at /home/cryo/tmux-logging"
    warn "Disabling tmux logging"
    enable_tmux_logging=false
fi

# Load AMC carrier configurations
if [ -z "$slot_cfgs" ]; then
    if [ -v "slots_in_configure_order" ]; then
	if [ -L /home/cryo/docker/smurf/current ]; then
	    info "Using slots_in_configure_order → /home/cryo/docker/smurf/current"
	    slots=("${slots_in_configure_order[@]}")
	    pyrogues=(`seq ${#slots_in_configure_order[@]} | awk '{print "/home/cryo/docker/smurf/current"}' | tr '\n' ' '`)
	else
	    error "slots_in_configure_order defined but /home/cryo/docker/smurf/current missing"
	    exit 1
	fi
    else
	if [ -z "$rfsoc_cfgs" ]; then
	    error "No slot_cfgs, slots_in_configure_order, or rfsoc_cfgs defined — nothing to do"
	    exit 1
	fi
    fi
else
    if [ -v "slots_in_configure_order" ]; then
	error "Both slot_cfgs and slots_in_configure_order defined — pick one"
	exit 1
    fi

    slots=( $(awk '{print $1}' <<< "$slot_cfgs") )
    pyrogues=( $(awk '{print $2}' <<< "$slot_cfgs") )
    pysmurf_cfgs=( $(awk '{print $3}' <<< "$slot_cfgs") )
fi

# Load RFSoC configurations
if [ ! -z "$rfsoc_cfgs" ]; then
    rfsoc_slots=( $(awk '{print $1}' <<< "$rfsoc_cfgs") )
    rfsoc_pyrogues=( $(awk '{print $2}' <<< "$rfsoc_cfgs") )
    rfsoc_pysmurf_cfgs=( $(awk '{print $3}' <<< "$rfsoc_cfgs") )
fi

# Print summary
if [ ${#slots[@]} -gt 0 ]; then
    printf "  ${DIM}AMC slots:${RESET}  ${BOLD}%s${RESET}\n" "${slots[*]}"
fi
if [ ${#rfsoc_slots[@]} -gt 0 ]; then
    printf "  ${DIM}RFSoC:${RESET}      ${BOLD}%s${RESET}\n" "${rfsoc_slots[*]}"
fi
printf "\n"

###############################################################################
# Cleanup
###############################################################################

header "CLEANUP"

# Set crate fans to full speed
if [ ${#slots[@]} -gt 0 ] && [ "$set_crate_fans_to_full" = true ] ; then
    info "Setting crate fans to max (level ${max_fan_level})"
    ssh root@${shelfmanager} "clia minfanlevel ${max_fan_level}; clia setfanlevel all ${max_fan_level}"
    sleep 2
fi

# Kill existing tmux session
info "Killing existing ${BOLD}${tmux_session_name}${RESET} tmux session"
tmux kill-session -t ${tmux_session_name} 2>/dev/null
sleep 0.1
info "Creating new ${BOLD}${tmux_session_name}${RESET} tmux session"
tmux new-session -d -s ${tmux_session_name}

if [[ "$enable_tmux_logging" = true ]]; then
    mkdir -vp /data/smurf_data/tmux_logs >/dev/null 2>&1
    tmux set -g @logging-path "/data/smurf_data/tmux_logs"
fi

# Stop pyrogue servers only for configured slots
for ((i=0; i<${#slots[@]}; ++i)); do
    slot=${slots[i]}
    pyrogue=${pyrogues[i]}
    if docker ps --filter "name=^smurf_server_s${slot}$" --format '{{.ID}}' | grep -q .; then
        stop_pyrogue $slot $pyrogue
    else
        dim "No running docker for slot ${slot}, skipping"
    fi
done

for ((i=0; i<${#rfsoc_slots[@]}; ++i)); do
    slot=${rfsoc_slots[i]}
    pyrogue=${rfsoc_pyrogues[i]}
    if docker ps --filter "name=^smurf_server_s${slot}$" --format '{{.ID}}' | grep -q .; then
        stop_pyrogue $slot $pyrogue
    else
        dim "No running docker for slot ${slot}, skipping"
    fi
done
cd $cpwd

# Check timing master
if [ "$using_timing_master" = true ] ; then
    matching_dockers tpg_ioc
    if [ "$?" = "1" ]; then
	success "tpg_ioc docker is running"
    else
	error "tpg_ioc docker is down — must start timing master first"
	exit 1
    fi
fi

###############################################################################
# Reboot
###############################################################################

if [ "$reboot" = true ] ; then

    header "REBOOT"

    # Reboot AMC carriers
    if [ ! -z "$slot_cfgs" ]; then
	deactivatecmd=""
	activatecmd=""
	for slot in ${slots[@]}; do
	    deactivatecmd="$deactivatecmd clia deactivate board ${slot};"
	    activatecmd="$activatecmd clia activate board ${slot};"
	done

	info "Deactivating carrier(s): ${BOLD}${slots[*]}${RESET}"
	ssh root@${shelfmanager} "$deactivatecmd"

	info "Waiting 5s before re-activation"
	sleep 5

	info "Activating carrier(s): ${BOLD}${slots[*]}${RESET}"
	ssh root@${shelfmanager} "$activatecmd"
    fi

    # Reboot RFSoCs
    if [ ! -z "$rfsoc_cfgs" ]; then
       warn "RFSoC reboot not yet supported"
    fi

fi

###############################################################################
# Configure carriers and RFSoCs
###############################################################################

header "SETUP"

all_slots=("${slots[@]}" "${rfsoc_slots[@]}")
all_pyrogues=("${pyrogues[@]}" "${rfsoc_pyrogues[@]}")
all_pysmurf_cfgs=("${pysmurf_cfgs[@]}" "${rfsoc_pysmurf_cfgs[@]}")

info "Bringing up ${BOLD}${#all_slots[@]}${RESET} slot(s): ${BOLD}${all_slots[*]}${RESET}"
printf "\n"

setup_complete=false
completion_status=7
declare -a slot_status=( $(for slot in ${slots[@]}; do echo 0; done) $(for rfsoc_slot in ${rfsoc_slots[@]}; do echo 1; done) )
setup_loop_cadence_sec=1

while [[ "${setup_complete}" = false ]] ; do
    for ((slot_idx=0; slot_idx<${#all_slots[@]}; ++slot_idx)); do
	slot=${all_slots[slot_idx]}
	pyrogue=${all_pyrogues[slot_idx]}
	pysmurf_cfg=${all_pysmurf_cfgs[slot_idx]}

	if [ "${slot_status[${slot_idx}]}" = "0" ]; then
	    cd $cpwd
	    ping_carrier -q 10.0.${crate_id}.$((${slot}+100))
	    slot_status[$slot_idx]=$?
	fi

	if [ "${slot_status[${slot_idx}]}" = "1" ]; then
	    start_slot_tmux_and_pyrogue ${slot} ${pyrogue}
	    slot_status[$slot_idx]=2
	fi

	if [ "${slot_status[${slot_idx}]}" = "2" ]; then
	    if is_slot_pyrogue_up ${slot}; then
		slot_status[$slot_idx]=3;
	    fi
	fi

	if [ "${slot_status[${slot_idx}]}" = "3" ]; then
	    if is_slot_server_up ${slot}; then
		slot_status[$slot_idx]=4;
	    fi
	fi

	if [ "${slot_status[${slot_idx}]}" = "4" ]; then
	    start_slot_pysmurf ${slot} ${pysmurf_cfg}
	    slot_status[$slot_idx]=5;
	    if [ "${configure_pysmurf}" = false ]; then
		slot_status[$slot_idx]=7;
	    fi
	fi

	if [ "${slot_status[${slot_idx}]}" = "5" ]; then
	    run_pysmurf_setup ${slot}
	    slot_status[$slot_idx]=6
	fi

	if [ "${slot_status[${slot_idx}]}" = "6" ]; then
	    if is_slot_pysmurf_setup_complete ${slot}; then
		slot_status[$slot_idx]=7;
	    fi
	fi

	# check if complete
	status_summary=(`echo ${slot_status[@]} | tr ' ' '\n' | sort | uniq`)
	if [[ "${#status_summary[@]}" = "1" && "${status_summary[0]}" = "${completion_status}" ]] ; then
	    setup_complete=true
	fi
    done

    # Print colored status line
    slot_status_line all_slots slot_status
    printf "\n"

    sleep ${setup_loop_cadence_sec}
done
printf "\r\033[K"

###############################################################################
# Post-setup
###############################################################################

if [ "$run_thermal_test" = true ] ; then
    header "THERMAL TEST"
    info "Launching thermal test in tmux window 8"
    tmux new-window -t ${tmux_session_name}:8
    tmux rename-window -t ${tmux_session_name}:8 tests
    tmux send-keys -t ${tmux_session_name}:8 'cd '${pysmurf} C-m
    tmux send-keys -t ${tmux_session_name}:8 'ipython3 -i pysmurf/'${thermal_test_script}' '`echo ${slots[@]} | tr ' ' ,` C-m
fi

if [ ! -z "$script_to_run" ]; then
    header "POST-SETUP SCRIPT"
    info "Running ${BOLD}${script_to_run}${RESET} on all slots"
    for slot in ${slots[@]}; do
	tmux send-keys -t ${tmux_session_name}:${slot} 'exec(open("'$script_to_run'").read())' C-m
    done
fi

###############################################################################
# Done
###############################################################################

_hammer_end=`date +%s`
_elapsed=$((_hammer_end - _hammer_start))

printf "\n"
printf "  ${BGREEN}✓ SMURFHAMMER COMPLETE${RESET}\n"
printf "  ${DIM}Elapsed: %dm %ds   Slots: %s${RESET}\n" \
       "$((_elapsed/60))" "$((_elapsed%60))" "${all_slots[*]}"
printf "  ${DIM}tmux attach -t %s${RESET}\n" "${tmux_session_name}"
printf "\n"

if [ "$attach_at_end" = true ] ; then
    tmux attach -t ${tmux_session_name}
fi

# Rarely used for some lofi debugging at Stanford.
if [ "$screenshot_signal_analyzer" = true ] ; then
    wid=`wmctrl -l | grep 171.64.108.28 | awk '{print $1}'`
    wmctrl -a 171.64.108.28
    import -window ${wid} /home/cryo/shawn/${ctime}_signal_analyzer.png
fi
