#!/usr/bin/env bash

########################
# Function definitions #
########################

# Import function from common script
. server_common.sh

# Usage message
usage()
{
    echo "Start the SMuRF server on a specific board."
    echo ""
    echo "usage: ${script_name} [-S|--shelfmanager <shelfmanager_name> -N|--slot <slot_number>]"
    echo "                      [-a|--addr <FPGA_IP>] [-D|--no-check-fw] [-g|--gui] <pyrogue_server-args>"
    echo "    -S|--shelfmanager <shelfmanager_name> : ATCA shelfmanager node name or IP address. Must be used with -N."
    echo "    -N|--slot         <slot_number>       : ATCA crate slot number. Must be used with -S."
    echo "    -a|--addr         <FPGA_IP>           : FPGA IP address. If defined, -S and -N are ignored."
    echo "    -c|--comm-type    <comm_type>         : Communication type ('eth' or 'pcie'). Default is 'eth'."
    echo "    -D|--no-check-fw                      : Disable FPGA version checking."
    echo "    -E|--disable-hw-detect                : Disable hardware type auto detection."
    echo "    -H|--hard-boot                        : Do a hard boot: reboot the FPGA and load default configuration."
    echo "    -h|--help                             : Show this message."
    echo "    <pyrogue_server_args> are passed to the SMuRF pyrogue server. "
    echo ""
    echo "If -a if not defined, then -S and -N must both be defined, and the FPGA IP address will be automatically calculated from the crate ID and slot number."
    echo "If -a if defined, -S and -N are ignored."
    echo
    echo "The script will by default check if the firmware githash read from the FPGA via IPMI is the same of the found in the MCS file name."
    echo "If they don't match, then the MCS file will be loaded into the FPGA. If this happens, the FPGA will be rebooted."
    echo "This checking can be disabled with -D. The checking will also be disabled if -a is used instead of -S and -N."
    echo
    echo "The script will try to auto-detect the type of hardware, and automatically generate server startup arguments based on the hardware type."
    echo "Currently, this script only detects the type of carrier board, and uses the '--enable-em22xx' option when the carrier is a Gen2, version >= C03."
    echo "This hardware auto-detection can be disabled using the option '-E|--disable-hw-detect'. This detection will also be disabled if -a is used instead"
    echo "of -S and -N. The user should provided the appropriate startup arguments based on the hardware type."
    echo
    echo "The script will look for a zip file under '${fw_top_dir}'. If found, it will be passed with the argument -z to the next startup script."
    echo "If not zip file is found, the script will then look for a local checked out repository in the same location; If found, the python directories"
    echo "under it will be added to PYTHONPATH."
    echo
    echo "The option '-H|--hard-boot' can be used to request a hard boot. During this boot mode, the FPGA is rebooted by deactivating and activating the carrier"
    echo "board before starting the pyrogue server, and the default configuration file is loaded during the pyrogue server booting process."
    echo
    echo "All other arguments are passed verbatim to the next startup script."
    echo ""
    exit 1
}

#############
# Main body #
#############

# Read inputs arguments
while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -S|--shelfmanager)
    shelfmanager="$2"
    shift
    ;;
    -N|--slot)
    slot="$2"
    shift
    ;;
    -D|--no-check-fw)
    no_check_fw=1
    ;;
    -E|--disable-hw-detect)
    disable_hw_detect=1
    ;;
    -a|--addr)
    fpga_ip="$2"
    shift
    ;;
    -c|--comm-type)
    comm_type="$2"
    shift
    ;;
    -H|--hard-boot)
    hard_boot=1
    shift
    ;;
    -s|--server)
    ;;
    -h|--help)
    usage
    ;;
    *)
    args="${args} $key"
    ;;
esac
shift
done

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

# Call the appropriate server startup script depending on the communication type
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    echo
    /usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}
else
    echo "Staring the server using PCIe communication..."
    echo
    /usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}
fi

