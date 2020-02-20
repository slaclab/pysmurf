slots=(2 3)

datafile=`find /data/smurf_data/westpak_thermal_testing_Jan2020/ -printf '%T+ %p\n' | sort -r | head -1 | awk '{print $2}'`

for slot in ${slots[@]}; do
    gnuplot -c plot_temperatures_continuous.gnuplot ${datafile} ${slot} &
done
