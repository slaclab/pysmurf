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
    -e           Open startup config in editor (default: emacs, or $EDITOR)
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

while getopts ":hei:c:t" opt; do
    case ${opt} in
      h )
          usage
          ;;
      e )
	  edit_config=true
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

if [ "$edit_config" = true ]; then
    ${EDITOR:-emacs} "$startup_cfg"
    exit 0
fi

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

# Style the tmux session — Smurf blue theme
tmux set -t ${tmux_session_name} status on
tmux set -t ${tmux_session_name} status-position bottom
tmux set -t ${tmux_session_name} status-interval 5
tmux set -t ${tmux_session_name} status-style "bg=#0d1117,fg=#58a6ff"
tmux set -t ${tmux_session_name} status-left-length 30
tmux set -t ${tmux_session_name} status-right-length 60
tmux set -t ${tmux_session_name} status-left "#[bg=#1f6feb,fg=#c9d1d9,bold] ░ ${tmux_session_name} #[bg=#0d1117,fg=#1f6feb]▓▒░"
tmux set -t ${tmux_session_name} status-right "#[fg=#264d73]│ #[fg=#79c0ff]%H:%M #[fg=#264d73]│ #[fg=#388bfd]%m-%d "
tmux set -t ${tmux_session_name} window-status-format "#[fg=#264d73] #I:#W "
tmux set -t ${tmux_session_name} window-status-current-format "#[bg=#161b22,fg=#58a6ff,bold] #I:#W "
tmux set -t ${tmux_session_name} window-status-separator ""
tmux set -t ${tmux_session_name} pane-border-style "fg=#1b3a5c"
tmux set -t ${tmux_session_name} pane-active-border-style "fg=#388bfd"
tmux set -t ${tmux_session_name} message-style "bg=#0d1117,fg=#79c0ff"
tmux set -t ${tmux_session_name} mode-style "bg=#1b3a5c,fg=#79c0ff"
tmux rename-window -t ${tmux_session_name}:0 "main"

if [[ "$enable_tmux_logging" = true ]]; then
    mkdir -vp /data/smurf_data/tmux_logs >/dev/null 2>&1
    tmux set -g @logging-path "/data/smurf_data/tmux_logs"
fi

# Stop server and client containers for configured slots
for ((i=0; i<${#slots[@]}; ++i)); do
    slot=${slots[i]}
    pyrogue=${pyrogues[i]}
    if docker ps -a --filter "name=^smurf_server_s${slot}$" --format '{{.ID}}' | grep -q . || \
       docker ps -a --filter "name=^pysmurf_s${slot}$" --format '{{.ID}}' | grep -q .; then
        info "Stopping containers for slot ${BOLD}${slot}${RESET}"
        docker rm -f "smurf_server_s${slot}" "pysmurf_s${slot}" 2>/dev/null
    else
        dim "No running docker for slot ${slot}, skipping"
    fi
done

for ((i=0; i<${#rfsoc_slots[@]}; ++i)); do
    slot=${rfsoc_slots[i]}
    pyrogue=${rfsoc_pyrogues[i]}
    if docker ps -a --filter "name=^smurf_server_s${slot}$" --format '{{.ID}}' | grep -q . || \
       docker ps -a --filter "name=^pysmurf_s${slot}$" --format '{{.ID}}' | grep -q .; then
        info "Stopping containers for slot ${BOLD}${slot}${RESET}"
        docker rm -f "smurf_server_s${slot}" "pysmurf_s${slot}" 2>/dev/null
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

# Launch all slots in parallel, each running the full setup sequence
declare -a _slot_pids=()
for ((i=0; i<${#all_slots[@]}; ++i)); do
    setup_slot ${all_slots[i]} ${all_pyrogues[i]} "${all_pysmurf_cfgs[i]}" &
    _slot_pids+=($!)
done

# Wait for all slots to finish
_any_failed=false
for pid in "${_slot_pids[@]}"; do
    if ! wait $pid; then
        _any_failed=true
    fi
done

if $_any_failed; then
    error "One or more slots failed during setup"
fi

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
