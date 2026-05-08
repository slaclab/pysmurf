import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rc('font',family='serif')
plt.rc('font',size=14)

# assuming you're already set up in a reasonable state
# no need to deal with configuring anything in cryo

tone_powers = [13] # I think we decided this was correct?
att_ucs = [0] # coarse loop just to see if things are very different
att_dcs = [0] # same

bands = [0,1,2,3,4,5,6,7]
#sampchan = 256 # this is basically in the middle of the band
chans_on = np.random.choice(510,250,replace=False) # just randomly choose half
chans_on = np.append(chans_on,511)
sampchan = 511 
nsamp = 2**22 # I think this is about 1/2 second of data? 
nperseg = 2**20

for tone_power in tone_powers:
    for att_uc in att_ucs:
        for att_dc in att_dcs:
            for band in bands:
                S.band_off(band) # turn everything off first
                S.set_att_uc(band, att_uc)
                S.set_att_dc(band, att_dc)

                for chan in chans_on:
                    S.set_amplitude_scale_channel(band,chan,tone_power)
                #S.set_amplitude_scale_channel(band, chan, tone_power)

            # restart the loop in order to make sure all the tones are on before datataking starts
            for band in bands:
                i,q,sync = S.take_debug_data(band=band,channel=sampchan,rf_iq=True,nsamp=nsamp)
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
                plt.legend(['I','Q'],loc='lower left')
                plt.ylim([-140,-70])
                plt.title(f'I/Q noise of Single Channel, 2000 on')
                plt.savefig(f'scratch/cyu/rsi_figs/loopback_manychan/iq_psd_b{band}ch{chan}_uc{att_uc}dc{att_dc}_power{tone_power}_2000x_nice.svg',bbox_inches='tight')
                plt.savefig(f'scratch/cyu/rsi_figs/loopback_manychan/iq_psd_b{band}ch{chan}_uc{att_uc}dc{att_dc}_power{tone_power}_2000x_nice.png',bbox_inches='tight')
                plt.close()

                #plt.figure() # plot timestream only
                #plt.plot(i[int(1e5):int(2e5)],alpha=0.8,label='digital i')
                #plt.plot(q[int(1e5):int(2e5)],alpha=0.8,label='digital q')
                #plt.xlabel('sample number')
                #plt.ylabel('voltage')
                #plt.title(f'I/Q timestream for b{band}ch{chan}, attuc{att_uc} attdc{att_dc}, tonepower {tone_power}, 2000 tones on')
                #plt.savefig(f'scratch/cyu/rsi_figs/loopback_manychan/iq_tod_b{band}ch{chan}_uc{att_uc}dc{att_dc}_power{tone_power}_2000x.png',bbox_inches='tight')
                #plt.close()


