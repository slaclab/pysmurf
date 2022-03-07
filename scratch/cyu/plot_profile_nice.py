import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

respi_1x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respi_1x.csv',delimiter=',')
respq_1x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respq_1x.csv',delimiter=',')
freqs_1x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/freqs_1x.csv',delimiter=',')
respi_2000x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respi_2000x.csv',delimiter=',')
respq_2000x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respq_2000x.csv',delimiter=',')
freqs_2000x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/freqs_2000x.csv',delimiter=',')

plt.figure(figsize=(12,8))
plt.plot(freqs_1x,respi_1x,'.',alpha=0.8,label='digital I')
plt.plot(freqs_1x,respq_1x,'.',alpha=0.8,label='digital Q')
plt.xlabel('Frequency [MHz]',fontsize=16)
plt.ylabel('Noise at 30kHz Offset [dBc/Hz]',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16,loc='upper left')
plt.title('Full SMuRF Band Noise, 1 Channel On',fontsize=20)
plt.ylim([-130,-75])
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_1x_nice.svg',bbox_inches='tight')
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_1x_nice.png',bbox_inches='tight')

plt.figure(figsize=(12,8))
plt.plot(freqs_2000x,respi_2000x,'.',alpha=0.8,label='digital I')
plt.plot(freqs_2000x,respq_2000x,'.',alpha=0.8,label='digital Q')
plt.xlabel('Frequency [MHz]',fontsize=16)
plt.ylabel('Noise at 30kHz Offset [dBc/Hz]',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16,loc='upper left')
plt.ylim([-130,-75])
plt.title('Full SMuRF Band Noise, 2000 Channels On',fontsize=20)
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_2000x_nice.svg',bbox_inches='tight')
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_2000x_nice.png',bbox_inches='tight')

plt.show()
