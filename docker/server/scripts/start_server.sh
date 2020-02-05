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
    echo "    -D|--no-check-fw                      : Disabled FPGA version checking."
    echo "    -H|--hard-boot                        : Do a hard boot: reboot the FPGA and load default configuration."
    echo "    -h|--help                             : Show this message."
    echo "    <pyrogue_server_args> are passed to the SMuRF pyrogue server. "
    echo ""
    echo "If -a if not defined, then -S and -N must both be defined, and the FPGA IP address will be automatically calculated from the crate ID and slot number."
    echo "If -a if defined, -S and -N are ignored."
    echo
    echo "The script will by default check if the firmware githash read from the FPGA via IPMI is the same of the found in the MCS file name."
    echo "If they don't match, then the MCS file will be loaded into the FPGA. If this happens, the FPGA will be rebooted."
    echo "This checking can be disabled with -D. The checking will also be disabled if -a is used instead of -S and -N. If "
    echo
    echo "The script will look for a zip file under '${fw_top_dir}'. If found, it will be passed with the argument -z to the next startup script."
    echo "If not zip file is found, the script will then look for a local checked out repository in the same location; If found, the python directories"
    echo "under it will be added to PYTHONPATH."
    echo
    echo "All other arguments are passed verbatim to the next startup script."
    echo ""
    exit 1
}

# Process and check input arguments
#
# During this call, several variables will be defined,
# depending on the input arguments:
# - shelfmanager: name of the ATCA crate shelfmanager,
# - slot: ATCA crate slot number,
# - no_check_fw: Flag to disable the automatic FW
#                version checking,
# - fpga_ip: FPGA IP address,
# - comm_type: communication type,
# - args: will contained the arguments to be passed
#         to the next startup script.
processArgs()
{
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

    # Verify mandatory parameters

    # Check communication type
    if [ -z ${comm_type+x} ]; then
        # If no type was selected, use 'eth' as default type
        comm_type='eth'
    else
        # Check if the communication type is invalid
        if [ ${comm_type} != 'eth' ] && [ ${comm_type} != 'pcie' ]; then
            echo "Invalid communication type!"
            usage
        fi
    fi

    # Check IP address or shelfmanager/slot number
    if [ -z ${fpga_ip+x} ]; then
        # If the IP address is not defined, shelfmanager and slot numebr must be defined

        if [ -z ${shelfmanager+x} ]; then
            echo "Shelfmanager not defined!"
            usage
        fi

        if [ -z ${slot+x} ]; then
            echo "Slot number not defined!"
            usage
        fi

        echo "IP address was not defined. It will be calculated automatically from the crate ID and slot number..."
        echo

	getIPAddr
    else
        echo "IP address was defined. Ignoring shelfmanager and slot number. FW version checking disabled."
        echo
        no_check_fw=1
    fi

    # Add the IP address to the SMuRF arguments
    args="${args} -a ${fpga_ip}"

    # If the slot number is defined, add the RSSI link number argument
    # which is needed if the PCIe card is used for communication
    if [ ${slot+x} ]; then
        # Verify that the slot number is in the range [2,7]
        if [ ${slot} -ge 2 -a ${slot} -le 7 ]; then
            args="${args} -l $((slot-2))"
        else
            echo "Invalid slot number! Must be a number between 2 and 7."
            exit 1
        fi
    fi
}

#############
# Main body #
#############

# Read input arguments
processArgs "$@"

# Look for pyrogue files
findPyrogueFiles

# Firmware version checking
if [ -z ${no_check_fw+x} ]; then
    checkFw
else
    echo "Check firmware disabled."
fi

# If a hard boot was request, reboot the FPGA and request the
# default configuration to be loaded
if  ! [ -z ${hard_boot+x} ]; then

    # Check if a new MCS was loaded. In that case there is not need
    # to reboot the FPGA, as it was already rebooted in the process
    # of loading a new firmware image.
    if [ -z ${mcs_loaded+x} ]; then
        rebootFPGA
    fi

    # Pass the -c argument to the SMuRF server to request the default
    # configuration to be loaded during the startup process
    args="${args} -c"
fi

# Call the appropriate server startup script depending on the communication type
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    /usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}
else
    echo "Staring the server using PCIe communication..."
    /usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}
fi

