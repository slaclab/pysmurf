#!/usr/bin/env bash

###############
# Definitions #
###############
# Shell PID
top_pid=$$

# This script name
script_name=$(basename $0)

# Firmware file location
fw_top_dir="/tmp/fw"

########################
# Function definitions #
########################

# Trap TERM signals and exit
trap "echo 'An ERROR was found. Check shelf manager & card state! Aborting...'; exit 1" TERM

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
    echo "    -h|--help                             : Show this message."
    echo "    <pyrogue_server_args> are passed to the SMuRF pyrogue server. "
    echo ""
    echo "If -a if not defined, then -S and -N must both be defined, and the FPGA IP address will be automatically calculated from the crate ID and slot number."
    echo "If -a if defined, -S and -N are ignored."
    echo
    echo "The script will by default check if the firmware githash read from the FPGA via IPMI is the same of the found in the MCS file name."
    echo "This checking can be disabled with -D. The checking will also be disabled if -a is used instead of -S and -N."
    echo
    echo "By default, the SMuRF server is tarted without a GUI. Use -g to start the server with a GUI."
    echo
    echo "All other arguments are passed verbatim to the next startup script."
    echo ""
    exit 1
}

getGitHashFW()
{
    local gh_inv
    local gh

    # Long githash (inverted)
    #gh_inv=$(ipmitool -I lan -H $SHELFMANAGER -t $IPMB -b 0 -A NONE raw 0x34 0x04 0xd0 0x14  2> /dev/null)
    # Short githash (inverted)
    gh_inv=$(ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x34 0x04 0xe0 0x04  2> /dev/null)

    if [ "$?" -ne 0 ]; then
        kill -s TERM ${top_pid}
    fi

    # Invert the string
    for c in ${gh_inv} ; do gh=${c}${gh} ; done

    # Return the short hash (7 bytes)
    echo ${gh} | cut -c 1-7
}

getGitHashMcs()
{
    local filename=$(basename $mcs_file_name)
    local gh=$(echo $filename | sed  -r 's/.+-+(.+).mcs.*/\1/')

    # Return the short hash (7 bytes)
    echo ${gh} | cut -c 1-7
}

getCrateId()
{
    local crate_id_str

    crate_id_str=$(ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x34 0x04 0xFD 0x02 2> /dev/null)

    if [ "$?" -ne 0 ]; then
        kill -s TERM ${top_pid}
    fi

    local crate_id=`printf %04X  $((0x$(echo $crate_id_str | awk '{ print $2$1 }')))`

    if [ -z ${crate_id} ]; then
        kill -s TERM ${top_pid}
    fi

    echo ${crate_id}
}

getFpgaIp()
{

    # Calculate FPGA IP subnet from the crate ID
    local subnet="10.$((0x${crate_id:0:2})).$((0x${crate_id:2:2}))"

    # Calculate FPGA IP last octect from the slot number
    local fpga_ip="${subnet}.$(expr 100 + $slot)"

    echo ${fpga_ip}
}

# Look for python directories in a local checkout of a repository.
# Python directories should match these patterns:
# - /tmp/fw/*/firmware/python/
# - /tmp/fw/*/firmware/submodules/*/python/
# All those found, will be added to PYTHONPATH
updatePythonPath()
{
    echo "Looking for a local checked out repository under '${fw_top_dir}'..."

    # Look for the python directories that match the patterns
    local python_dirs=( $(find ${fw_top_dir} -type d \
                          -regex "^${fw_top_dir}/[^/]+/firmware/python" -o \
                          -regex "^${fw_top_dir}/[^/]+/firmware/submodules/[^/]+/python") )

    # Check if any directory was found
    if [ ${#python_dirs[@]} -eq 0 ]; then
        # if nothing was found, just return without doing anything
        echo "Not python directories found"
    else
        # If directories were found,add them all to PYTHONPATH
        echo "The following python directories were found:"
        for d in ${python_dirs[@]}; do
            echo ${d}
            python_path=${d}:${python_path}
        done

        export PYTHONPATH=${python_path}${PYTHONPATH}
        echo "PYTHONPATH updated!"
    fi
}

#############
# Main body #
#############

# Verify inputs arguments
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

echo

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

    ipmb=$(expr 0128 + 2 \* $slot)

    echo "Reading Crate ID via IPMI..."
    crate_id=$(getCrateId)
    echo "Create ID: ${crate_id}"

    echo "Calculating FPGA IP address..."
    fpga_ip=$(getFpgaIp)
    echo "FPGA IP: ${fpga_ip}"

else
    echo "IP address was defined. Ignoring shelfmanager and slot number. FW version checking disabled."
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

# Look for a pyrogue zip file
echo "Looking for pyrogue zip file..."
pyrogue_file=$(find ${fw_top_dir} -maxdepth 1 -name *zip)
if [ ! -f "$pyrogue_file" ]; then
    echo "Pyrogue zip file not found!"

    # if not found, then look for a local checkout repository.
    updatePythonPath
else
    # If found, add it to the SMuRF arguments
    echo "Pyrogue zip file found: ${pyrogue_file}"
    args="${args} -z ${pyrogue_file}"
fi

# Firmware version checking
if [ -z ${no_check_fw+x} ]; then

    mcs_file=$(find ${fw_top_dir} -maxdepth 1 -name *mcs*)
    if [ ! -f "${mcs_file}" ]; then
        echo "MCS file not found!"
        exit 1
    fi

    mcs_file_name=$(basename ${mcs_file})
    echo ${mcs_file_name}

    echo "Reading FW Git Hash via IPMI..."
    fw_gh=$(getGitHashFW)
    echo "Firmware githash: '$fw_gh'"

    echo "Reading MCS file Git Hash..."
    mcs_gh=$(getGitHashMcs)
    echo "MCS file githash: '$mcs_gh'"

    if [ "${fw_gh}" == "${mcs_gh}" ]; then
        echo "They match..."
    else
        echo "They don't match. Loading image..."
        ProgramFPGA.bash -s $shelfmanager -n $slot -m $mcs_file
    fi

else
    echo "Check firmware disabled."
fi

# Call the appropriate server startup script depending on the communication type
if [ ${comm_type} == 'eth' ]; then
    echo "Staring the server using Ethernet communication..."
    /usr/local/src/pysmurf/server_scripts/cmb_eth.py  ${args}
else
    echo "Staring the server using PCIe communication..."
    /usr/local/src/pysmurf/server_scripts/cmb_pcie.py ${args}
fi

