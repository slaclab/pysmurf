#!/bin/bash

quiet=false

while getopts ":q" opt; do
  case ${opt} in
    q )
      quiet=true
     ;;
  esac
done
shift $((OPTIND -1))

if [ "$quiet" = true ] ; then
    timeout 0.2 ping -c 1 $1 >/dev/null && exit 1
    exit 0
else
    CYAN='\033[0;36m'
    BGREEN='\033[1;32m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'

    frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    i=0
    start=$(date +%s)

    printf "\033[?25l"
    trap 'printf "\033[?25h"' EXIT

    while ! timeout 0.2 ping -c 1 -n $1 &> /dev/null
    do
        elapsed=$(( $(date +%s) - start ))
        printf "\r\033[K  ${CYAN}%s${RESET} Waiting for ${BOLD}%s${RESET} ${DIM}(%ds)${RESET}" \
            "${frames[$i]}" "$1" "$elapsed" >&2
        i=$(( (i + 1) % ${#frames[@]} ))
        sleep 0.3
    done

    elapsed=$(( $(date +%s) - start ))
    printf "\r\033[K  ${BGREEN}✓${RESET} ${BOLD}%s${RESET} is online ${DIM}(%ds)${RESET}\n" "$1" "$elapsed"
    printf "\033[?25h"
fi
