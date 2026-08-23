import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

respi_1x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respi_1x.csv',delimiter=',')
respi_2000x = np.loadtxt('scratch/cyu/profile_loopback_data/uc6dc6/respi_2000x.csv',delimiter=',')

plt.figure(figsize=(8,6))
plt.hist(respi_1x,bins=50,alpha=0.5,color='red')
plt.hist(respi_2000x,bins=50,alpha=0.5,color='blue')
plt.xlabel('Noise at 30kHz offset [dBc/Hz]',fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.yticks(fontsize=14)
plt.title('1 vs 2000 Channel Loopback across 4GHz',fontsize=18)
plt.legend(['1 channel on','2000 channels on'],fontsize=14)
plt.xlim([-125,-90])
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_hist.svg',bbox_inches='tight')
plt.savefig('scratch/cyu/rsi_figs/profile_loopback_hist.png',bbox_inches='tight')
plt.show()
