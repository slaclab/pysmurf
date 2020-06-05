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
nperseg = 2**17
reset_rate_khzs = np.array([4, 10, 20, 40])
n_phi0s = np.array([4, 4, 4, 4])
lms_enable2 = False
lms_enable3 = False
lms_gain = 3
filter_order = 4

n_steps = len(reset_rate_khzs)

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
ff_nofr, pxx_nofr = signal.welch(d, fs=S.get_channel_frequency_mhz()*1.0E6,
    nperseg=nperseg)
idx = np.argsort(ff_nofr)
# plt.figure()
# plt.semilogy(f[idx]*1.0E-3, pxx[idx])
# plt.xlabel('Freq [kHz]')
# plt.show()

f = {}
df = {}
ff = {}
pxx = {}
for i in np.arange(n_steps):
    f[i], df[i], sync = S.tracking_setup(band,
        channel=channel, reset_rate_khz=reset_rate_khzs[i],
        fraction_full_scale=.5, make_plot=True, show_plot=False, nsamp=2**18,
        lms_gain=lms_gain, lms_freq_hz=None, meas_lms_freq=False,
        feedback_start_frac=.25, feedback_end_frac=.98, meas_flux_ramp_amp=True,
        n_phi0=n_phi0s[i], lms_enable2=lms_enable2, lms_enable3=lms_enable3)
    I, Q, sync = S.take_debug_data(band=band, channel=channel, rf_iq=True)
    d = I * 1.j*Q

    ff[i], pxx[i] = signal.welch(d, fs=S.get_channel_frequency_mhz()*1.0E6,
        nperseg=nperseg)

cm = plt.get_cmap('viridis')
plt.figure(figsize=(8,4.5))
for i in np.arange(n_steps):
    color = cm(i/n_steps)
    plt.semilogy(ff[i][idx], pxx[i][idx], color=color,
        label=f'{reset_rate_khzs[i]*n_phi0s[i]*1.0E-3} kHz')
plt.plot(ff_nofr[idx], pxx_nofr[idx], color='k', label='None')
plt.show()