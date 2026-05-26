import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rc('font',family='serif')
plt.rc('font',size=14)

# assuming you're already set up in a reasonable state
# no need to deal with configuring anything in cryo

#tone_powers = [12,13,14,15] # I think I want maximum s/n out of DAC but check
#att_ucs = [0,6,12,18,24] # coarse loop just to see if things are very different
#att_dcs = [0,6,12,18,24] # same

tone_powers = [13]
att_ucs = [6]
att_dcs = [6]

bands = [0,1,2,3,4,5,6,7]
chan = 504 # 256 is the middle; 504 is some offset
nsamp = 2**22 # I think this is about 1/2 second of data? 
nperseg = 2**19

for band in bands:
    for att_uc in att_ucs:
        for att_dc in att_dcs:
            for tone_power in tone_powers:
                S.set_att_uc(band, att_uc)
                S.set_att_dc(band, att_dc)

                S.set_amplitude_scale_channel(band, chan, tone_power)

                i,q,sync = S.take_debug_data(band=band,channel=chan,rf_iq=True,nsamp=nsamp)
                S.set_amplitude_scale_channel(band, chan, 0) # remember to turn it back off!!!
                ffi,pxxi = signal.welch(i,fs=S.get_channel_frequency_mhz()*1.e6,nperseg=nperseg)
                ffq,pxxq = signal.welch(q,fs=S.get_channel_frequency_mhz()*1.e6,nperseg=nperseg)

                # scale to dBc/Hz by the voltage magnitude
                magfac = np.mean(q)**2 + np.mean(i)**2
                pxxi_dbc = 10. * np.log10(pxxi/magfac)
                pxxq_dbc = 10. * np.log10(pxxq/magfac)

                plt.figure()
                plt.semilogx(ffi,pxxi_dbc,alpha=0.8,label='digital I')
                plt.semilogx(ffq,pxxq_dbc,alpha=0.8,label='digital Q')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Power Spectral Density [dBc/Hz]')
                plt.legend(['I','Q'],fontsize=14,loc='lower left')
                plt.ylim([-140,-70])
                plt.title(f'Digitial I/Q Noise of Single Channel, 1 on')
                plt.savefig(f'scratch/cyu/rsi_figs/loopback_singlechan/iq_psd_b{band}ch{chan}_uc{att_uc}dc{att_dc}_power{tone_power}_nice.svg',bbox_inches='tight')
                plt.savefig(f'scratch/cyu/rsi_figs/loopback_singlechan/iq_psd_b{band}ch{chan}_uc{att_uc}dc{att_dc}_power{tone_power}_nice.png',bbox_inches='tight')
                plt.close()


