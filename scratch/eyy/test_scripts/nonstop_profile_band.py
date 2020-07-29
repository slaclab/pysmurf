import numpy as np
import profile_band
import os


bands = np.arange(8)
epics_root = 'smurf_server_s5'
config_file = '/usr/local/src/pysmurf/cfg_files/rflab/experiment_rflab_thermal_testing_201907.cfg'
shelf_manager = 'shm-smrf-sp01'
loopback = True

output_dir = '/data/smurf_data/20200406/'

print(f'Config file is {config_file}')

while True:
    # Loop over bands
    for band in bands:
        html_path = profile_band.run(band, epics_root, config_file,
            shelf_manager, True, loopback=loopback,
            no_find_freq=True,
            no_setup_notches=True, no_band_off=True)
        f = open(os.path.join(output_dir, f'profile_band{band}.txt'), 'ab')
        np.savetxt(f, [html_path], fmt='%s')
        f.close()

