#!/usr/bin/env bash

###############
# Definitions #
###############

# This script name
script_name=$(basename $0)

########################
# Function definitions #
########################

# Usage message
usage()
{
    echo "Start the SMuRF client, pointing to an specific server."
    echo ""
    echo "usage: ${script_name} [-e|--epics <epics_prefix>] [-c|--config-file <config_file>] [-h|--help]"
    echo "    -e|--epics <epics_prefix>      : Sets the EPICS PV name prefix (defaults to 'smurf_server')."
    echo "    -c|--config-file <config_file> : Path to the configuration path"
    echo "    -h|--help                      : Show this message."
    echo ""
}

#############
# Main body #
#############

# Verify inputs arguments
while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -e|--epics)
    epics_prefix="$2"
    shift
    ;;
    -c|--config-file)
    config_file="$2"
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "ERROR: Unknown argument..."
    usage
    exit 1
    ;;
esac
shift
done

echo

# The epics prefix is defined in the environmental variable 'EPICS_PREFIX'
# and it is set to 'smurf_server' by default in the Dockerfile.
# If a new prefix is passed as an argument, override the environmental variable.
if [ ! -z ${epics_prefix+x} ]; then
    echo "Setting EPCIS_PREFIX environmental variable to ${epics_prefix}..."
    export EPICS_PREFIX=${epics_prefix}
fi

# Set the environmental variable 'CONFIG_FILE' to the passed argument
if [ ! -z ${config_file+x} ]; then
    if [ ! -f ${config_file} ]; then
        echo "ERROR: Configuration file '${config_file}' not found!"
    else
        echo "Setting CONFIG_FILE environmental variable to ${config_file}..."
        export CONFIG_FILE=${config_file}
    fi
fi

echo "Starting the ipython session"
ipython3 -i /usr/local/src/pysmurf_utilities/pysmurf_startup.py