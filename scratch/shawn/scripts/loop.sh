rm -v loop_full_band_resps.txt

for i in `seq 1 50`; do
    shawnhammer
    echo "Waiting 15 seconds before next reboot."
    sleep 15
done
