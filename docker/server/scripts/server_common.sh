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

# Trap TERM signals and exit
trap "echo 'An ERROR was found. Check shelf manager & card state! Aborting...'; exit 1" TERM

########################
# Function definitions #
########################

getGitHashFW()
{
    local gh_inv
    local gh

    # Long githash (inverted)
    #gh_inv=$(ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x34 0x04 0xd0 0x14  2> /dev/null)
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
    printf "Looking for local python directories... "

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
            echo "  ${d}"
            python_path=${d}:${python_path}
        done

        export PYTHONPATH=${python_path}${PYTHONPATH}
        echo "  PYTHONPATH updated!"
    fi
}

# Generate IP address from the shelfmanager name
# and slot number
getIPAddr()
{
    ipmb=$(expr 0128 + 2 \* $slot)

    printf "Reading Crate ID via IPMI...            "
    crate_id=$(getCrateId)
    echo "Create ID: ${crate_id}"

    printf "Calculating FPGA IP address...          "
    fpga_ip=$(getFpgaIp)
    echo "FPGA IP: ${fpga_ip}"
}

# Check if firmware in FPGA matches MCS file
checkFw()
{
    printf "Looking for mcs file...                 "
    mcs_file=$(find ${fw_top_dir} -maxdepth 1 -name *mcs*)
    if [ ! -f "${mcs_file}" ]; then
        echo "MCS file not found!"
        exit 1
    fi

    mcs_file_name=$(basename ${mcs_file})
    echo "Mcs file found: ${mcs_file_name}"

    printf "Reading FW Git Hash via IPMI...         "
    fw_gh=$(getGitHashFW)
    echo "Firmware githash: '$fw_gh'"

    printf "Reading MCS file Git Hash...            "
    mcs_gh=$(getGitHashMcs)
    echo "MCS file githash: '$mcs_gh'"

    if [ "${fw_gh}" == "${mcs_gh}" ]; then
        echo "They match!"
    else
        echo "They don't match. Loading image..."
        ProgramFPGA.bash -s $shelfmanager -n $slot -m $mcs_file
    fi
}

# Look for pyrogue classes. First, look for a zip
# file. If not found , then look for local checked
# out python directories.
findPyrogueFiles()
{
    # Look for a pyrogue zip file
    printf "Looking for pyrogue zip file...         "
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
}
