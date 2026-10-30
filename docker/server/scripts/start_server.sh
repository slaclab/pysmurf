#!/usr/bin/env bash
#
# The server container's entry point. Everything the launcher used to work out
# here -- the FPGA's address from the crate id, the PCIe lane, the board
# population and its register defaults, the firmware package and whether the
# FPGA runs it, a hard boot -- is the Python server's own now: name the crate
# slot with -S/--shelfmanager and -N/--slot and it reads the rest from the
# shelf manager. Every option is passed through unchanged, with one
# translation kept for the deployments that spell it: -c/--comm-type
# <eth|pcie|emu> is the server's --transport <eth|pcie|emulation>. (The server's
# own -c is --configure, which is why the translation has to happen here.)

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--comm-type)
            case "$2" in
                eth)  args+=(--transport eth) ;;
                pcie) args+=(--transport pcie) ;;
                emu)  args+=(--transport emulation) ;;
                *)    echo "start_server.sh: -c/--comm-type takes eth, pcie or emu, not '$2'" >&2; exit 2 ;;
            esac
            shift 2 ;;
        *)
            args+=("$1"); shift ;;
    esac
done

# With the GUI requested, make sure a display exists.
if printf '%s\n' "${args[@]}" | grep -qxE '(-g|--gui)'; then
    if [ -z "${DISPLAY}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
        echo "No working display found. Starting Xvfb..."
        Xvfb :99 -screen 0 1920x1080x24 &
        export DISPLAY=:99
        for i in $(seq 1 10); do
            if xdpyinfo -display :99 >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
        echo "Xvfb started on display ${DISPLAY}"
    fi
fi

echo "python3 -m pysmurf.core.server ${args[*]}"
exec python3 -m pysmurf.core.server "${args[@]}"
