#!/bin/bash
# [$1==slots in a comma delimited list]

IFS=", " read -r -a slots <<< "$1"
echo ${slots[@]}

datafile=`find /data/smurf_data/simonsobs_first10carriers_thermal_testing_Feb2020/ -printf '%T+ %p\n' | sort -r | head -1 | awk '{print $2}'`

for slot in ${slots[@]}; do
    gnuplot -c plot_temperatures_continuous.gnuplot ${datafile} ${slot} &
done
