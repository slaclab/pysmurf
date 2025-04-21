#rm -v /data/smurf_data/smurf_loop.log

for i in `seq 1 5`; do
    shawnhammer
    echo "Waiting 60 seconds before next reboot."
    sleep 60
done
