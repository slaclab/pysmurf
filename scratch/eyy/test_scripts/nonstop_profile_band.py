import numpy as np
import profile_band
import os

bands = np.array([2, 3])
epics_root = 'smurf_server_s5'
config_file = '/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg'
shelf_manager = 'shm-smrf-sp01'

output_dir = '/data/smurf_data/20200406/'

while True:
    # Loop over bands
    for band in bands:
        output_dir = profile_band.run(band, epics_root, config_file,
        f = open(os.path.join(output_dir, f'profile_band{band}.txt'), 'ab')
        np.savetxt(f, output_dir)
        f.close()