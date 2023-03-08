# Debugs find frequency script and plotting
import sys
sys.path.append('../../../../')
import numpy as np
import os
import matplotlib.pylab as plt

def plot_find_freq(path,ct,color,label,band_center_mhz=0):
    freq = np.loadtxt(os.path.join(path,f'{ct}_amp_sweep_freq.txt'))
    freq+=band_center_mhz
    resp = np.genfromtxt(os.path.join(path,f'{ct}_amp_sweep_resp.txt'), dtype=complex)
    subband = np.arange(512)
    
    for i, sb in enumerate(subband):
        if i==0:
            # so we only plot legend once
            plt.plot(freq[sb,:], np.abs(resp[sb,:]), linestyle='-', marker='.', markersize=4,
                     color=color, alpha=0.75,label=f'{ct} : {label}')
        else:
            plt.plot(freq[sb,:], np.abs(resp[sb,:]), linestyle='-', marker='.', markersize=4,
                     color=color, alpha=0.75)            

plt.figure(figsize=(10,4))

# 100mK?
path='/data/smurf_data/20230228/1677624034/outputs/'
ct=1677624885
plot_find_freq(path,ct,color='c', label='100mK?',band_center_mhz=480)

# 300mK
path='/data/smurf_data/20230228/1677625941/outputs/'
ct=1677626181
plot_find_freq(path,ct,color='r', label='300mK',band_center_mhz=480)

# 350mK
path='/data/smurf_data/20230228/1677625941/outputs/'
ct=1677627876
plot_find_freq(path,ct,color='g', label='350mK',band_center_mhz=480)

plt.title("find_freq response")
plt.xlabel("Frequency offset (MHz)")
plt.ylabel("Normalized Amplitude")

plt.legend(loc='upper right',fontsize=12)

plt.show()
