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

# Call the appropriate server startup script depending on the communication type
# and pass the list of arguments 'args'.
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}"
else
    echo "Staring the server using PCIe communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}"
fi

echo ${cmd}
${cmd}
