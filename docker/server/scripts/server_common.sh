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

rebootFPGA()
{
    local retry_max=10
    local retry_delay=10

    printf "Sending reboot command to FPGA...       "
    ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x2C 0x0A 0 0 2 0 &> /dev/null
    sleep 1
    ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x2C 0x0A 0 0 1 0 &> /dev/null
    echo "Done"

    printf "Waiting for FPGA to boot...             "
    # Wait until FPGA boots
    for i in $(seq 1 $retry_max); do
        sleep $retry_delay
        local bsi_state=$(ipmitool -I lan -H $shelfmanager -t $ipmb -b 0 -A NONE raw 0x34 0xF4 2> /dev/null | awk '{print $1}')
        if [ "$?" -eq 0 ] && [ $bsi_state -eq 3 ]; then
	    local done=1
            break
        fi
    done

    if [ -z $done ]; then
        echo "FPGA didn't boot after $(($retry_max*$retry_delay)) seconds. Aborting..."
        echo
        exit 1
    else
        echo "FPGA booted after $((i*$retry_delay)) seconds"
    fi
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

# Get FPGA IP address
getFpgaIpAddr()
{
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

        ipmb=$(expr 0128 + 2 \* $slot)

        printf "Reading Crate ID via IPMI...            "
        crate_id=$(getCrateId)
        echo "Create ID: ${crate_id}"

        printf "Calculating FPGA IP address...          "
        fpga_ip=$(getFpgaIp)
        echo "FPGA IP: ${fpga_ip}"

    else
        echo "IP address was defined. Ignoring shelfmanager and slot number. FW version checking disabled."
        echo
        no_check_fw=1
    fi

    # Add the IP address to the SMuRF arguments
    args="${args} -a ${fpga_ip}"
}

# Validate the selected slot number
validateSlotNumber()
{
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

# Check if firmware in FPGA matches MCS file
checkFW()
{
    # Check if the firmware checking is disabled
    if [ -z ${no_check_fw+x} ]; then

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

            # Set a flag indicating a new MCS was loaded
            mcs_loaded=1
        fi

    else
        echo "Check firmware disabled."
    fi
}

# Validate the communication type selected
validateCommType()
{
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
}

# Do a hard boot, if requested
hardBoot()
{
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
