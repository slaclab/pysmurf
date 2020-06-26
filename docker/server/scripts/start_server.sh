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

# Validate the selected communication type
validateCommType

# Validate the selected slot number
validateSlotNumber

# Get FPGA IP address
getFpgaIpAddr

# Look for pyrogue files
findPyrogueFiles

# Firmware version checking
checkFW

# Do a hard boot, if requested
hardBoot

# Auto-detect hardware type
## Detect type of AMCs, and get specific server startup arguments
## for each specific type and add them to the list of arguments
detect_amc_board amcs_args
args+=" ${amcs_args}"

## Detect type of carrier, and get specific server startup arguments
## for each specific type and add them to the list of arguments
detect_carrier_board carrier_args
args+=" ${carrier_args}"

echo

# Call the appropriate server startup script depending on the communication type
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}"
else
    echo "Staring the server using PCIe communication..."
    cmd="/usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}"
fi

echo ${cmd}
${cmd}
