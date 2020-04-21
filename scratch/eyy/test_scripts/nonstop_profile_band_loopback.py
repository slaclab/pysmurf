import numpy as np
import profile_band
import os

#############################
# Notes:
#
# April 20 2020
# -------------
# Run on smurf-srv11
# loopback
# Only run on slot 3 for now.
#############################

bands = np.arange(4)
epics_root = 'smurf_server_s3'
config_file = '/data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg'
shelf_manager = 'shm-smrf-sp01'

output_dir = '/data/smurf_data/20200420/'

# For simplicity - handle loopback here
loopback = True
if loopback:
    no_setup_notches = True
    no_find_freq = True
else:
    no_setup_notches = False
    no_find_freq = False


while True:
    # Loop over bands
    for band in bands:
        html_path = profile_band.run(band, epics_root, config_file,
            shelf_manager, True, no_setup_notches=no_setup_notches,
            no_find_freq=no_find_freq)
        f = open(os.path.join(output_dir, f'profile_band{band}.txt'), 'ab')
        np.savetxt(f, [html_path], fmt='%s')
