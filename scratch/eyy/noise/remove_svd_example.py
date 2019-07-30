import pysmurf
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import os

S = pysmurf.SmurfControl(cfg_file='/Users/edwardyoung/Documents/umux/pysmurf/cfg_files/experiment_k2umux.cfg',
    data_dir='.', setup=False, offline=True, no_dir=True)

datadir = '/Users/edwardyoung/Desktop/dat'
# filename = '1548453806.dat'
filename = '1557340156.dat'
datafile = os.path.join(datadir, filename)

t,d,m = S.read_stream_data(datafile)
_, n_res = np.shape(np.where(m!=-1))

d *= S.pA_per_phi0/(2*np.pi)  # phase to pA

# Take SVD
u, s, vh = S.noise_svd(d, m)

# ============================================================
# Totally done with SVD-ing. Everything below just diagnostics
# ============================================================


# Make summary plots
S.plot_svd_modes(vh, show_plot=True)
S.plot_svd_summary(u, s, show_plot=True)
plt.title('{}'.format(filename))

modes = 4
d_clean = S.remove_svd(d, m, u, s, vh, modes=modes)

# Look at one of the resontors
res_num = 300
plt.figure()
plt.plot(d[res_num], label='raw')
plt.plot(d_clean[res_num], label='subtract')
plt.legend()

nperseg = 2**12
dirty = np.zeros((n_res,3))
clean = np.zeros((n_res,3))

for i in np.arange(n_res):
    f, pxx = signal.welch(d[i], nperseg=nperseg, fs=180)
    pxx = np.sqrt(pxx)
    popt, pcov, f_fit, pxx_fit = S.analyze_psd(f, pxx)
    dirty[i] = popt


    f, pxx = signal.welch(d_clean[i], nperseg=nperseg, fs=180)
    pxx = np.sqrt(pxx)
    popt, pcov, f_fit, pxx_fit = S.analyze_psd(f, pxx)
    clean[i] = popt

plt.figure(figsize=(8,4.5))
plt.plot(dirty[:,0], '.', label='original')
plt.plot(clean[:,0], 'x', label='- {} modes'.format(modes))
plt.legend()
plt.xlabel('Res Num')
plt.ylabel('White Noise [pA/rtHz]')
plt.yscale('log')
plt.ylim((10,1000))
plt.title(filename)
