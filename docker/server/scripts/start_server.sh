#!/usr/bin/env bash

########################
# Function definitions #
########################

# Import functions from common script
. server_common.sh

# The common functions can be override here, if needed.

#############
# Main body #
#############

# Parse the inputs arguments.
# The list of extra arguments will be store in 'args'
arg_parser args "$@"

# Call the initialization routines.
# The list of extra arguments will be store in 'extra_args' and it
# will be added to the list arguments 'args'.
initialize extra_args
args+=" ${extra_args}"

echo

# If the GUI flag (-g or --gui) is present, ensure a display is available.
# Start Xvfb if DISPLAY is not set or the X server is not reachable.
if echo "${args}" | grep -qE '(^|\s)(-g|--gui)(\s|$)'; then
    if [ -z "${DISPLAY}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
        echo "No working display found. Starting Xvfb..."
        Xvfb :99 -screen 0 1920x1080x24 &
        export DISPLAY=:99
        # Wait for Xvfb to be ready
        for i in $(seq 1 10); do
            if xdpyinfo -display :99 >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
        echo "Xvfb started on display ${DISPLAY}"
    fi
fi

# Start the server. The transport is the communication type; everything else in
# 'args' is passed through to the server's own command line.
case ${comm_type} in
    eth) transport=eth ;;
    emu) transport=emulation ;;
    *)   transport=pcie ;;
esac
echo "Starting the server over ${transport}..."
cmd="python3 -m pysmurf.core.server --transport ${transport} ${args}"
echo ${cmd}
exec ${cmd}
