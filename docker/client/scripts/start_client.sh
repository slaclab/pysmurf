#!/usr/bin/env bash
# Start the SMuRF client: an ipython session with a SmurfControl connected to a server.
#
# The client reaches a server over its rogue ZMQ sockets, so the one thing it needs is
# where the server listens: a host and port, or a crate slot for a server on this host.
# A configuration file is optional; without one the client reads the configuration
# back from a server that has been set up.

script_name=$(basename "$0")
startup=/usr/local/src/pysmurf_utilities/pysmurf_startup.py

usage()
{
    echo "Start the SMuRF client, connected to a server."
    echo ""
    echo "usage: ${script_name} [-t|--target <target>] [-c|--config-file <config_file>] [-h|--help]"
    echo "    -t|--target <target>           : The server: 'host:port', or 'crate:<slot>' for a server on this"
    echo "                                     host serving an ATCA slot (defaults to 'localhost:9000')."
    echo "    -c|--config-file <config_file> : A configuration file (YAML) for the client. Without one, the"
    echo "                                     configuration is read back from a server that has been set up."
    echo "    -h|--help                      : Show this message."
    echo ""
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -t|--target)
    [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage; exit 1; }
    export SMURF_TARGET="$2"
    shift
    ;;
    -c|--config-file)
    [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage; exit 1; }
    if [ ! -f "$2" ]; then
        echo "ERROR: Configuration file '$2' not found!" >&2
        exit 1
    fi
    export CONFIG_FILE="$2"
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "ERROR: Unknown argument '$1'" >&2
    usage
    exit 1
    ;;
esac
shift
done

echo "Starting the ipython session (server ${SMURF_TARGET:-localhost:9000}${CONFIG_FILE:+, configuration ${CONFIG_FILE}})"
exec ipython3 -i "${startup}"
