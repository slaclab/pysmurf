#!/bin/bash
# [$1==slots in a comma delimited list]
# [$2==path to log file]

IFS=", " read -r -a slots <<< "$1"
echo ${slots[@]}

datafile=$2

for slot in ${slots[@]}; do
    gnuplot -c plot_temperatures_continuous.gnuplot ${datafile} ${slot} &
done
