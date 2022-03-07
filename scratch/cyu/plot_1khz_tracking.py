# standardize plotting of tracking bandwidth
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

filename = '/data/smurf_data/20211207/1638904355/outputs/1638926676.dat'

t,d,m,b = S.read_stream_data(filename,return_tes_bias=True)

idxs_on = np.where(m!=-1)[0]
idxs_band3 = np.where(idxs_on==3)[0]


fig,ax = plt.subplots(2,1,figsize=(8,10))
ax[0].plot(t[200:400]*1e6/(4.e3),1e6*(d[idxs_band3[46],200:400]-np.mean(d[idxs_band3[46],200:400]))*9e-6 / (2*np.pi),'.-') 
ax[0].plot(t[200:400]*1e6/(4.e3),1e6*(b[0,200:400]-np.mean(b[0,200:400]))*2*S._rtm_slow_dac_bit_to_volt/S.bias_line_resistance,'.-') 
ax[0].set_xlabel('time [us]',fontsize=16) 
ax[0].set_ylabel('current [uA]',fontsize=16) 


ff,pxx = welch(d,fs=4000)
ff2,pxx2 = welch(b[0],fs=4000)

ax[1].loglog(ff,np.sqrt(pxx[idxs_band3[46]] / np.max(pxx[idxs_band3[46]])))
ax[1].loglog(ff2,np.sqrt(pxx2/np.max(pxx2)))
ax[1].set_xlabel('frequency [Hz]',fontsize=16)
ax[1].set_ylabel('PSD [dBc/rtHz]',fontsize=16)
ax[1].legend(['measured','input'],fontsize=16)
ax[0].set_title('Resolving 1kHz sine on Detector Bias',fontsize=18)
plt.savefig('scratch/cyu/rsi_figs/sinewave.svg',bbox_inches='tight')
plt.savefig('scratch/cyu/rsi_figs/sinewave.png',bbox_inches='tight')
plt.show()

