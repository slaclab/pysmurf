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

# Call the appropriate server startup script depending on the communication type
# and pass the list of arguments 'args'.
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}"
elif [ ${comm_type} == 'emu' ]; then
    echo "Staring the server using Emulation..."
    cmd="/usr/local/src/pysmurf/server_scripts/emulate.py  ${args}"
else
    echo "Staring the server using PCIe communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}"
fi

echo ${cmd}
${cmd}
