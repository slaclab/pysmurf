#!/bin/bash

## Stripped down version for parallelization which returns exit value
## 0 if ping fails or 1 if ping succeeds
#timeout 0.2 ping -c 1 $1 >/dev/null && exit 1
#exit 0

printf "%s" "waiting for $1 ..."
while ! timeout 0.2 ping -c 1 -n $1 &> /dev/null
do
    printf "%c" "."
done
printf "\n%s\n"  "$1 is online"
