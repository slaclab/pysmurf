import numpy as np
import glob
import os
import matplotlib.pyplot as plt

fileroot = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180920/' + \
    '1537461840/outputs'

files = glob.glob(os.path.join(fileroot, '*freq_full_band_resp.txt'))
files = np.array(['/home/common/data/cpu-b000-hp01/cryo_data/data2/20180920/1537461840/outputs/1537461850_freq_full_band_resp.txt'])
n_files = len(files)
freq = np.loadtxt(files[0])
idx = np.argsort(freq)
freq = freq[idx]
n_pts = len(freq)

resp = np.zeros((n_files, n_pts), dtype=complex)

for i, f in enumerate(files):
    resp[i] = (np.loadtxt(f.replace('freq', 'real')) + \
        1.j*np.loadtxt(f.replace('freq', 'imag')))[idx]

resp = np.ravel(resp)

# fig, ax = plt.subplots(2, sharex=True)
# resp_mean = np.mean(resp, axis=0)
# for i in np.arange(n_files):
#     ax[0].semilogy(freq, np.abs(resp[i]))
#     ax[1].plot(freq, np.abs(resp[i] - resp_mean))

import pysmurf
S = pysmurf.SmurfControl()

grad_loc = S.find_peak(freq, resp)
fig, ax = plt.subplots(1)
ax.plot(freq, np.abs(resp), '-bD', markevery=grad_loc)
ax.set_yscale('log')

