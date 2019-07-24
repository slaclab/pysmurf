#!/bin/bash

quiet=false

# Parse options to the `pip` command
while getopts ":q" opt; do
  case ${opt} in
    q )
      quiet=true
     ;;
  esac
done
shift $((OPTIND -1))

if [ "$quiet" = true ] ; then
    # Stripped down version for parallelization which returns exit value
    ## 0 if ping fails or 1 if ping succeeds
    timeout 0.2 ping -c 1 $1 >/dev/null && exit 1
    exit 0
else 
    printf "%s" "waiting for $1 ..."
    while ! timeout 0.2 ping -c 1 -n $1 &> /dev/null
    do
	printf "%c" "."
    done
    printf "\n%s\n"  "$1 is online"
fi
