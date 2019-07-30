#!/bin/bash
printf "%s" "waiting for $1 ..."
while ! timeout 0.2 ping -c 1 -n $1 &> /dev/null
do
    printf "%c" "."
done
printf "\n%s\n"  "$1 is online"
