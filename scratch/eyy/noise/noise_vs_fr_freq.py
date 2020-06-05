import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

####
# Assumes you've already setup the system.
####

### System Configuration ###
epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')
tune_file = '/data/smurf_data/tune/1590781150_tune.npy'

### Function variables ###
band = 2
channel = 443

# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

S.load_tune(filename=tune_file)
S.relock(band)
S.band_off(band)
S.set_amplitude_scale_channel(band, channel, 12)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

I, Q, sync = S.take_debug_data(band=band, channel=channel, rf_iq=True,
    IQstream=False)
d = I * 1.j*Q
f, pxx = signal.welch(d, fs=S.get_channel_frequency_mhz()*1.0E6,
    nperseg=len(d))
idx = np.argsort(f)
plt.figure()
plt.semilogy(f[idx]*1.0E-3, pxx[idx])
plt.xlabel('Freq [kHz]')
plt.show()