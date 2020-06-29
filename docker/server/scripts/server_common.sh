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

# Usage message
usage()
{
    echo "Start the SMuRF server on a specific board."
    echo ""
    echo "usage: ${script_name} [-S|--shelfmanager <shelfmanager_name> -N|--slot <slot_number>]"
    echo "                       [-a|--addr <FPGA_IP>] [-D|--no-check-fw] <pyrogue_server-args>"
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

# Argument parser. It will generate variables used by the rest of functions (like
# 'shelfmanager', 'slot', etc) depending on the inputs arguments. Also, it will
# return a list of all the extra arguments which were not parsed by this function.
# This list can then be passed to the next script.
# The first argument indicates the variable to store the result list of extra argument.
# The second argument is the list of inputs arguments.
arg_parser()
{
    # The first argument points to a variable name to store the resulting argument list
    local __result_args=$1

    # Shift so that we point now to the list of inputs arguments
    shift

    # This will hold the list of extra arguments
    local __extra_args

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
            -h|--help)
            usage
            ;;
            *)
            __extra_args="${__extra_args} $key"
            ;;
        esac
        shift
    done

    # Write the result to the defined output variable
    eval $__result_args="'${__extra_args}'"
}

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
    local bsi_state

    printf "Sending reboot command to FPGA...                 "
    ipmitool -I lan -H ${shelfmanager} -t ${ipmb} -b 0 -A NONE raw 0x2C 0x0A 0 0 2 0 &> /dev/null

    # Verify IPMI errors
    if [ "$?" -ne 0 ]; then
        kill -s TERM ${top_pid}
        exit
    fi

    sleep 5

    ipmitool -I lan -H ${shelfmanager} -t ${ipmb} -b 0 -A NONE raw 0x2C 0x0A 0 0 1 0 &> /dev/null

    # Verify IPMI errors
    if [ "$?" -ne 0 ]; then
        kill -s TERM ${top_pid}
        exit
    fi

    echo "Done."

    printf "Waiting for FPGA to boot...                       "

    # Wait until FPGA boots
    for i in $(seq 1 ${retry_max}); do

        sleep ${retry_delay}
        bsi_state=$(ipmitool -I lan -H ${shelfmanager} -t ${ipmb} -b 0 -A NONE raw 0x34 0xF4 2> /dev/null | awk '{print $1}')

        # Verify IPMI errors
        if [ "$?" -eq 0 ] && [ ${bsi_state} -eq 3 ]; then
            local ready_fpga=1
            break
        fi

    done

    if [ -z ${ready_fpga+x} ]; then
        echo "FPGA didn't boot after $((${retry_max}*${retry_delay})) seconds. Aborting..."
        echo
        kill -s TERM ${top_pid}
        exit
    else
        echo "FPGA booted after $((i*${retry_delay})) seconds"
    fi

    printf "Waiting for FPGA's ETH to come up...              "

    # Wait until FPGA's ETH is ready
    for i in $(seq 1 ${retry_max}); do

        if /bin/ping -c 2 ${fpga_ip} &> /dev/null ; then
           local ready_eth=1
           break
        else
           sleep ${retry_delay}
        fi

    done

    if [ -z ${ready_eth+x} ]; then
        echo "FPGA's ETH didn't come up after $((${retry_max}*${retry_delay})) seconds. Aborting..."
        echo
        kill -s TERM ${top_pid}
        exit
    else
        echo "FPGA's ETH came up after $((i*${retry_delay})) seconds."
    fi
}

getFpgaIp()
{
    # Calculate FPGA IP subnet from the crate ID
    local subnet="10.$((0x${crate_id:0:2})).$((0x${crate_id:2:2}))"

    # Calculate FPGA IP last octet from the slot number
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
    printf "Looking for local python directories...           "

    # Look for the python directories that match the patterns
    local python_dirs=( $(find ${fw_top_dir} -type d \
                          -regex "^${fw_top_dir}/[^/]+/firmware/python" -o \
                          -regex "^${fw_top_dir}/[^/]+/firmware/submodules/[^/]+/python") )

    # Check if any directory was found
    if [ ${#python_dirs[@]} -eq 0 ]; then
        # if nothing was found, just return without doing anything
        echo "Not python directories found."
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
        # If the IP address is not defined, shelfmanager and slot number must be defined

        if [ -z ${shelfmanager+x} ]; then
            echo "Shelfmanager not defined!."
            usage
        fi

        if [ -z ${slot+x} ]; then
            echo "Slot number not defined!."
            usage
        fi

        echo "IP address was not defined. It will be calculated automatically from the crate ID and slot number..."

        ipmb=$(expr 0128 + 2 \* $slot)

        printf "Reading Crate ID via IPMI...                      "
        crate_id=$(getCrateId)
        echo "Create ID: ${crate_id}."

        printf "Calculating FPGA IP address...                    "
        fpga_ip=$(getFpgaIp)
        echo "FPGA IP: ${fpga_ip}."

    else
        # We  need the shelfmanager and slot number in order to get information
        # via IPMI, which we do to get the FW version and to auto detect the HW type.
        # So, when the IP address is used instead, disable these two features.
        echo "IP address was defined. Ignoring shelfmanager and slot number."
        echo "FW version checking and HW type auto  detection disabled."
        echo
        no_check_fw=1
        disable_hw_detect=1
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

        printf "Looking for mcs file...                           "
        mcs_file=$(find ${fw_top_dir} -maxdepth 1 -name *mcs*)
        if [ ! -f "${mcs_file}" ]; then
            echo "MCS file not found!."
            exit 1
        fi

        mcs_file_name=$(basename ${mcs_file})
        echo "Mcs file found: ${mcs_file_name}"

        printf "Reading FW Git Hash via IPMI...                   "
        fw_gh=$(getGitHashFW)
        echo "Firmware githash: '$fw_gh'."

        printf "Reading MCS file Git Hash...                      "
        mcs_gh=$(getGitHashMcs)
        printf "MCS file githash: '$mcs_gh'. "

        if [ "${fw_gh}" == "${mcs_gh}" ]; then
            echo "They match!."
        else
            echo "They don't match."
            echo "Loading image..."
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
            echo "Invalid communication type!."
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
    printf "Looking for pyrogue zip file...                   "
    pyrogue_file=$(find ${fw_top_dir} -maxdepth 1 -name *zip)
    if [ ! -f "$pyrogue_file" ]; then
        echo "Pyrogue zip file not found!."

        # if not found, then look for a local checkout repository.
        updatePythonPath
    else
        # If found, add it to the SMuRF arguments
        echo "Pyrogue zip file found: '${pyrogue_file}'."
        args="${args} -z ${pyrogue_file}"
    fi
}

# Detect the type of carrier board
# and generate a list of according server input arguments
# The first argument indicates the a variable to store the resulting argument list
detect_carrier_board()
{
    # Check if the hardware type auto-detection is disabled
    if [ -z ${disable_hw_detect+x} ]; then

        # The first argument a variable name to store the resulting argument list
        local __result_args=$1

        # Definitions
        ## AMC carrier board part numbers
        local carrier_gen1_pn="PC_379_396_01"
        local carrier_gen2_pn="PC_379_396_38"

        echo "Auto-detecting type and version of AMC carrier board:"

        # Get the AMC carrier board part number
        local pn_str=$(cba_fru_init --dump ${shelfmanager}/${slot} | grep -Po "Board Part Number\s+:\s+\K.+")

        # Check if a board is present in this slot
        printf "Checking if board is present...                   "
        if [ -z ${pn_str} ]; then
            echo "Board not present."
            echo
            echo "The carrier board is not present, or it contains an empty 'Board Part Number' string. Aborting!"
            echo "Please install a supported carrier board, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
            echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
            echo "file using the '--defaults <path_to_defaults_file>' argument."
            echo
            exit 1
        else
            echo "Board present."
        fi

        # Verify if the part number is correct
        printf "Verifying the part number is supported...         "
        local gen1=$(echo ${pn_str} | grep -o ${carrier_gen1_pn})
        local gen2=$(echo ${pn_str} | grep -o ${carrier_gen2_pn})

        if [ -z ${gen1} ] && [ -z ${gen2} ]; then
            echo "Part number ${pn_str} not supported."
            echo
            echo "Unsupported carrier board, or with bad 'Board Part Number' string, was found. Aborting!"
            echo "Please install a supported carrier board, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
            echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
            echo "file using the '--defaults <path_to_defaults_file>' argument."
            echo
            exit 1
        else
            echo "Part number ${pn_str} supported."
        fi

        printf "Type of carrier is:                               "
        if [ ${gen1} ]; then
            echo "Gen1."
        else
            echo "Gen2."
        fi

        # Extract version from the part number string
        if [ ${gen1} ]; then
            local ver_str=$(echo ${pn_str} | grep -Po "${carrier_gen1_pn}_C\K[0-9]{2}")
        else
            local ver_str=$(echo ${pn_str} | grep -Po "${carrier_gen2_pn}_C\K[0-9]{2}")
        fi

        # Verify if we extracted a version number string
        printf "Verifying board version...                        "
        if [ -z ${ver_str} ]; then
            echo "Version not found in the part number string."
            echo
            echo "Unsupported carrier board, or with bad 'Board Part Number' string, was found. Aborting!"
            echo "Please install a supported carrier board, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
            echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
            echo "file using the '--defaults <path_to_defaults_file>' argument."
            echo
            exit 1
        else
            echo "c${ver_str}."
        fi

        # Assemble the arguments and default file name
        local args=""

        # For Gen2 C03 carriers, we need to activate the EM22xx device via an input argument
        if [ ${gen2} ] && [ ${ver_str} -ge 03 ]; then
            args+="--enable-em22xx"
        fi

        # Print the final list of auto-generated arguments
        echo "Final list of generated arguments:                '${args}'"
        echo "Done!."

        # Write the result to the defined variable
        eval $__result_args="'${args}'"
    else
        echo "Auto-detection of type and version of AMC carrier board disabled."
    fi
}

# Detect presence, type, and version of AMC boards,
# and generated a list of according server input arguments.
# The first argument indicates the variable to store the resulting argument list
detect_amc_board()
{
    # Check if the hardware type auto-detection is disabled
    if [ -z ${disable_hw_detect+x} ]; then

        # The first argument points to a variable name to store the resulting argument list
        local __result_args=$1

        # Definitions
        ## SMuRF AMC board part number
        local smurf_pn="PC_379_396_30"
        ## Path to default files
        local defaults_path="/tmp/fw/smurf_cfg/defaults"

        # Array variables, one entry for each bay:
        ## Band type
        declare -a local band_bay=("" "")
        ## Version
        declare -a local ver_bay=("" "")

        echo "Auto-detecting type and version of AMC boards:"
        # Loop to read board in both slots
        for i in $(seq 0 1); do

            # The bay number are 0 and 2, so need to multiply the index (0,1) by 2
            local bay=$((i*2))
            echo "- Reading board on bay ${bay}:"

            # Get the AMC board part number
            local pn_str=$(cba_amc_init --dump ${shelfmanager}/${slot}/${bay} | grep -Po "Board Part Number\s+:\s+\K.+")

            # Check if a board is present in this slot
            printf "  Checking if board is present...                 "
            if [ -z ${pn_str} ]; then
                echo "Board not present."
                continue
            else
                echo "Board present."
            fi

            # Verify if the part number is correct
            printf "  Verifying the part number is supported...       "
            local supported=$(echo ${pn_str} | grep -o ${smurf_pn})

            if [ -z ${supported} ]; then
                echo "Part number ${pn_str} not supported."
                echo
                echo "Unsupported AMC board, or with bad 'Board Part Number' string, was found. Aborting!"
                echo "Please install supported AMC boards, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
                echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
                echo "file using the '--defaults <path_to_defaults_file>' argument."
                echo
                exit 1
            else
                echo "Part number ${pn_str} supported."
            fi

            # Extract type and version from the part number string
            local type_str=$(echo ${pn_str} | grep -Po "${smurf_pn}_C[0-9]{2}_\KA[0-9]{2}")
            local ver_str=$(echo ${pn_str} | grep -Po "${smurf_pn}_C\K[0-9]{2}")

            # Verify if we extracted a version number string
            printf "  Verifying board version...                      "
            if [ -z ${ver_str} ]; then
                echo "Version not found in the part number string."
                echo
                echo "Unsupported AMC board, or with bad 'Board Part Number' string, was found. Aborting!"
                echo "Please install supported AMC boards, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
                echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
                echo "file using the '--defaults <path_to_defaults_file>' argument."
                echo
                exit 1
            else
                echo "c${ver_str}."
            fi

            # Verify is we extracted a type string
            printf "  Verifying board type...                         "
            if [ -z ${type_str} ]; then
                echo "Board type not found in the part number string."
                echo
                echo "Unsupported AMC board, or with bad 'Board Part Number' string, was found. Aborting!"
                echo "Please install supported AMC boards, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
                echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
                echo "file using the '--defaults <path_to_defaults_file>' argument."
                echo
                exit 1
            else
                 printf "${type_str}. "
            fi

            # Verify if the type is supported
            if [ ${type_str} == "A01" ]; then
                echo "This is a LB board."
                band_bay[$i]="lb"
            elif [ ${type_str} == "A02" ]; then
                echo "This is a HB board."
                band_bay[$i]="hb"
            else
                echo "Board type not supported."
                echo
                echo "Unsupported AMC board, or with bad 'Board Part Number' string, was found. Aborting!"
                echo "Please install supported AMC boards, or disable the hardware auto-detection by passing the flag '--disable-hw-detect'."
                echo "You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults"
                echo "file using the '--defaults <path_to_defaults_file>' argument."
                echo
                exit 1
            fi

            # Now that we verify that the type is correct, and that we extracted
            # a version number, write that version into the version array. We will
            # use this array to determine if the board is present and supported
            # (and empty string here means there is not board, or that is it not supported).
            ver_bay[${i}]="c${ver_str}"

        done

        # Checking if both boards have the same version
        if [ ${ver_bay[0]} ] && [ ${ver_bay[1]} ]; then
            printf "Boards present in both bays. Checking versions... "
            if [ ${ver_bay[0]} == ${ver_bay[1]} ]; then
                echo "Versions match."
            else
                echo "Version don't match: ${ver_bay[0]} != ${ver_bay[1]}."
                echo
                echo "Different board versions in the same carrier are not supported. Aborting!"
                echo "Please install AMC boards of the same version, or disable the hardware auto-detection by passing the flag"
                echo "'--disable-hw-detect'. You will need to manually pass the appropriate '--disable-bay[0,1]' flags as well"
                echo "as pointing to the correct defaults file using the '--defaults <path_to_defaults_file>' argument."
                echo
                exit 1
            fi
        fi

        # Get the version of the board that is present.
        # If both are present, we already check at this point
        # that they have the same version
        if [ ${ver_bay[0]} ]; then
            local ver=${ver_bay[0]}
        else
            local ver=${ver_bay[1]}
        fi

        # Assemble the arguments and default file name
        local args=""
        local defaults_file_name="${defaults_path}/defaults_${ver}"

        for i in $(seq 0 1); do
            defaults_file_name+="_"
            if [ ! ${ver_bay[$i]} ]; then
                args+="--disable-bay$i "
                defaults_file_name+="none"
            else
                defaults_file_name+="${band_bay[$i]}"
            fi
        done

        defaults_file_name+=".yml"

        echo "Defaults file name:                               ${defaults_file_name}."

        # Check if default file exist
        printf "Verifying is default file exist...                "
        if [ ! -f ${defaults_file_name} ]; then
            echo "File not found."
            echo
            echo "The defaults file with auto-generated name was not found. Aborting!"
            echo "Please make sure a defaults file for this hardware configuration exists and that it follows the file naming convention."
            echo "Alternative, you could disable the hardware auto-detection by passing the flag '--disable-hw-detect'. You will need to"
            echo "manually pass the appropriate '--disable-bay[0,1]' flags as well as pointing to the correct defaults file using the"
            echo "'--defaults <path_to_defaults_file>' argument."
            echo
            exit 1
        else
            echo "File found!."
            # As the default file exist, add it to the list of arguments
            args+="-d ${defaults_file_name}"
        fi


        # Print the final list of auto-generated arguments
        echo "Final list of generated arguments:                '${args}'."
        echo "Done!."

        # Write the result to the defined variable
        eval $__result_args="'${args}'"
    else
         echo "Auto-detection of presence, type and version of AMC boards disabled."
    fi
}

# This function calls all the initialization routines. It returns a list of extra
# arguments which are generated by some if the routines that auto-detect hardware versions.
# The first argument indicates the variable to store the result list of extra argument.
initialize()
{
    # The first argument points to a variable name to store the resulting argument list
    local __result_args=$1

    # This will hold the list of extra arguments
    local __extra_args

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
    local __extar_amcs_args
    detect_amc_board __extar_amcs_args
    __extra_args+=" ${__extar_amcs_args}"

    ## Detect type of carrier, and get specific server startup arguments
    ## for each specific type and add them to the list of arguments
    local __extar_carrier_args
    detect_carrier_board __extar_carrier_args
    __extra_args+=" ${__extar_carrier_args}"

    # Write the result to the defined output variable
    eval $__result_args="'${__extra_args}'"
}