import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

####
# Assumes you've already setup the system.
####

### System Configuration ###
datafile_file = os.path.join('/data/smurf_data/20200529/1590773672/outputs/',
    '1590774028_compare_tracking_noise.npy')
fs = 200
nperseg = 2**12
bins_min = np.array([.2, .5, 1, 3, 10])
bins_max = np.array([.5, 1, 3, 10, 30])
n_bins = len(bins_min)

epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')


# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

datafiles = np.load(datafile_file, allow_pickle=True).item()

pxx_all = {}
bin_vals_all = {}

for kk in datafiles:
    datafile = datafiles[kk]
    # Load data
    print(datafile)
    t, d, m = S.read_stream_data(datafile)
    d *= S.pA_per_phi0/2/np.pi

    bands, channels = np.where(m != -1)
    n_chan = len(bands)

    pxx = np.zeros((n_chan, nperseg//2+1))
    bin_vals = np.zeros((n_chan, n_bins))
    for i, (b, ch) in enumerate(zip(bands, channels)):
        key = m[b, ch]
        f, pxx[i] = signal.welch(d[key], fs=fs, nperseg=nperseg)

        for j, (bmin, bmax) in enumerate(zip(bins_min, bins_max)):
            idx = np.where(np.logical_and(f > bmin, f < bmax))
            bin_vals[i, j] = np.median(pxx[key][idx])

    bin_vals_all[kk] = bin_vals
    pxx_all[kk] = pxx