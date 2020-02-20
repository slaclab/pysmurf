datafile=`find /data/smurf_data/rflab_thermal_testing_swh_July2019/ -printf '%T+ %p\n' | sort -r | head -1 | awk '{print $2}'`

gnuplot -c plot_temperatures_continuous.gnuplot ${datafile}
