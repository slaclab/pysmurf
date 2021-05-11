import pysmurf.client
import numpy as np
import scipy.signal as signal
import os
import glob
import matplotlib.pyplot as plt

# What to run
preprocess = False
psd = True

# Data
output_dir = '/data/smurf_data/princeton_srv15/20210219/1613776439/outputs'
fs = 1000
pA_per_phi0 = 9E6

datafiles = np.load(os.path.join(output_dir, 'out_dict2.npy'),
    allow_pickle=True).item()
keys = list(datafiles.keys())

S = pysmurf.client.SmurfControl(offline=True)

# Preprocess all the data
if preprocess:
    for k in keys:
        print(f'Working on {k}')
        fn = os.path.basename(datafiles[k])

        # Extract values
        reset_rate = int(k[5])
        nphi0 = int(k[-2:])

        t, d, m = S.read_stream_data(os.path.join(output_dir, fn))

        bands, channels = np.where(m != -1)

        for b, ch in zip(bands, channels):
            idx = m[b, ch]
            np.save(os.path.join(output_dir,
                f'b{b}ch{ch:03}_rr{reset_rate}_nphi{nphi0:02}'), d[idx])

# Load the data, take PSD, then plot.
# T his is currently hardcoded to only work for band 2. Need to update
# code if we want to change.
if psd:
    datafiles = glob.glob(os.path.join(output_dir, 'b*ch*.npy'))
    bands = np.zeros(len(datafiles), dtype=int)
    channels = np.zeros_like(bands)
    reset_rates = np.zeros_like(bands)
    nphi0s = np.zeros_like(bands)

    # Load the bands, channels
    for i, d in enumerate(datafiles):
        df = os.path.basename(d)
        bands[i] = int(df[1])
        channels[i] = int(df[4:7])
        reset_rates[i] = int(df[10])
        nphi0s[i] = int(df[16:18])

    # Find the unique channels
    unique_channels = np.unique(channels)

    for ch in unique_channels:
        fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)
        idx = np.where(channels == ch)[0]

        for i in idx:
            d = np.load(os.path.join(output_dir, datafiles[i]))

            # Convert to pA
            d *= pA_per_phi0/(2*np.pi)
            f, pxx = signal.welch(d, fs=fs, nperseg=2**15)
            pxx = np.sqrt(pxx)

            ax[0].loglog(f, pxx,
                label=f'rr {reset_rates[i]}'+ r'$N \phi_0$' + f' {nphi0s[i]}')
            phi0_rate = reset_rates[i] * nphi0s[i]

            # Calculate the median in the signal band.
            psd_idx = np.where(np.logical_and(f>.1, f<10))
            med_pxx = np.median(pxx[psd_idx])

            ax[1].plot(phi0_rate, med_pxx, '.')

        ax[0].legend()
        ax[0].set_ylim(100,5000)
        ax[0].set_xlabel('Freq [Hz]')
        ax[0].set_ylabel('Noise PSD [pA/rtHz]')
        ax[1].set_xlabel(f'$\phi_0$ rate [kHz]')
        plt.tight_layout()

        fig.suptitle(f'b{bands[i]}ch{ch:03}')

        # Save the data
        plt.savefig(os.path.join(output_dir, f'b{2}ch{ch:03}.png'),
            bbox_inches='tight')
        plt.close()