if [ -z $1 ]
then
    echo "Must provide num_averages as argument!"
    exit 1
fi

# must be run in the utils docker
num_averages=$1
slot=4

wait_to_read_cfg_sec=0.1
wait_btw_each_caput_sec=10

## Joe confirms that the procedure should be
#1) starting state ; userConfig[0]=6 ; userConfig[0] bits 1 and 2 high, all other bits low.
#2) enable streaming
#3) change smurf2mce.cfg file.
#4) toggle userConfig[0] bit 3 high.
#5) wait for config file to be read.
#6) toggle userConfig[0] bit 3 low.
#7) toggle userConfig[0] bit 2 low.

# make sure smurf2mce debugging is active
echo "Setting mcetransmitDebug->1"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:mcetransmitDebug 1' C-m
sleep ${wait_btw_each_caput_sec}

# enable streaming
echo "Setting enableStreaming->1"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:FpgaTopLevel:AppTop:AppCore:enableStreaming 1' C-m
sleep ${wait_btw_each_caput_sec}

# bits 0, 1, and 2 high, all others low
echo "Setting userConfig[0]->7"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:FpgaTopLevel:AppTop:AppCore:TimingHeader:userConfig[0] 6' C-m
sleep ${wait_btw_each_caput_sec}

sed -i "s/num_averages.*/num_averages ${num_averages}/g" /data/smurf2mce_config/smurf2mce.cfg
# not actually a caput, but whatever
sleep ${wait_btw_each_caput_sec}

# toggle bit 3
echo "Setting userConfig[0]->14"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:FpgaTopLevel:AppTop:AppCore:TimingHeader:userConfig[0] 14' C-m
sleep ${wait_btw_each_caput_sec}
echo "Waiting ${wait_to_read_cfg_sec} sec for config file read"
sleep ${wait_to_read_cfg_sec}
echo "Setting userConfig[0]->6"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:FpgaTopLevel:AppTop:AppCore:TimingHeader:userConfig[0] 6' C-m
sleep ${wait_btw_each_caput_sec}

# toggle bit 2 low to start writing to file
echo "Setting userConfig[0]->2"
tmux send-keys -t smurf:utils 'caput smurf_server_s'${slot}':AMCc:FpgaTopLevel:AppTop:AppCore:TimingHeader:userConfig[0] 2' C-m
sleep ${wait_btw_each_caput_sec}


