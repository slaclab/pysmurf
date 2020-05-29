import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os

####
# Assumes you've already setup the system.
####

### System Configuration ###
epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')

### Function variables ###
band = 2
subband = np.arange(50, 52)

reset_rate_khzs = np.array([4, 10, 15, 15])
n_phi0s = np.array([4, 4, 4, 6])
lms_enable2 = False
lms_enable3 = False
lms_gain = 7

# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

S.find_freq(band, subband, make_plot=True)
S.setup_notches(band, new_master_assignment=True)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

f = {}
df = {}

n_steps = len(reset_rate_khzs)

for i in np.arange(n_steps):
    f[i], df[i], sync = S.tracking_setup(band, reset_rate_khz=reset_rate_khzs[i],
        fraction_full_scale=.5, make_plot=False, nsamp=2**18, lms_gain=lms_gain,
        lms_freq_hz=None, meas_lms_freq=False, feedback_start_frac=.2,
        feedback_end_frac=.98, meas_flux_ramp_amp=True, n_phi0=n_phi0s[i],
        lms_enable2=lms_enable2, lms_enable3=lms_enable3)