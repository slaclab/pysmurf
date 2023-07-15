import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rc('font',family='serif')
plt.rc('font',size=14)

# assuming you're already set up in a reasonable state
# no need to deal with configuring anything in cryo

tone_powers = [13] # I think I want maximum s/n out of DAC but check
att_ucs = [12] # coarse loop just to see if things are very different
att_dcs = [12] # same

off = 30 # kHz offset to sample at

bands = [0,1,2,3,4,5,6,7]
#chan = 504 # 256 is the middle; 504 is some offset
nsamp = 2**20 # I think this is about 1/2 second of data? 
nperseg = 2**18
freqs = np.arange(-249.5,250,2) # fine loop now to match the 1000x resolution for posting. This will take basically all weekend.
f_idxs = np.arange(len(freqs))

for att_uc in att_ucs:
    for att_dc in att_dcs:
        for tone_power in tone_powers:
            plt.figure()
            bandvec_i = np.zeros((8*len(freqs),1))
            bandvec_q = np.zeros((8*len(freqs),1))
            for band in bands:
                print(f'working on band {band} power {tone_power} uc {att_uc} dc {att_dc}')

                # set the attenuators
                S.set_att_uc(band, att_uc)
                S.set_att_dc(band, att_dc)

                center_freq_mhz = 4250 + 500*band
                for fidx in f_idxs:
                    f = freqs[fidx]
                    try:
                        b,chan = S.set_fixed_tone(center_freq_mhz + f, tone_power)
 
                        i,q,sync = S.take_debug_data(band=band,channel=chan,rf_iq=True,nsamp=nsamp)
                        S.set_amplitude_scale_channel(b, chan, 0) # remember to turn it back off!!!
                        ffi,pxxi = signal.welch(i,fs=S.get_channel_frequency_mhz()*1.e6,nperseg=nperseg)
                        ffq,pxxq = signal.welch(q,fs=S.get_channel_frequency_mhz()*1.e6,nperseg=nperseg)

                        # scale to dBc/Hz by the voltage magnitude
                        magfac = np.mean(q)**2 + np.mean(i)**2
                        pxxi_dbc = 10. * np.log10(pxxi/magfac)
                        pxxq_dbc = 10. * np.log10(pxxq/magfac)
                        # get the index for offset, in kHz
                        freq_idx = np.where(ffi >= off*1e3)[0][0]
                        bandvec_i[len(freqs)*band + fidx] = pxxi_dbc[freq_idx]
                        bandvec_q[len(freqs)*band + fidx] = pxxq_dbc[freq_idx]

                    except AssertionError:
                        continue

                #plt.semilogx(ffi,pxxi_dbc,alpha=0.8,label='digital I')
                #plt.semilogx(ffq,pxxq_dbc,alpha=0.8,label='digital Q')
            xvec = np.hstack((4250+freqs,4750+freqs,5250+freqs,5750+freqs,6250+freqs,6750+freqs,7250+freqs,7750+freqs))
            plt.plot(xvec,bandvec_i,'.',alpha=0.8,label='digital I')
            plt.plot(xvec,bandvec_q,'.',alpha=0.8,label='digital Q')
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('Noise at 30kHz offset [dBc/Hz]')
            plt.title(f'I/Q noise of low band AMC, attuc{att_uc} attdc{att_dc}, tonepower {tone_power}')
            plt.savefig(f'scratch/cyu/rsi_figs/profile_singlechan/iq_psd_lowband_uc{att_uc}dc{att_dc}_power{tone_power}_offset{off}khz.png',bbox_inches='tight')
            plt.close()

            np.savetxt('freqs_1x.csv',xvec,delimiter=',')
            np.savetxt('respi_1x.csv',bandvec_i,delimiter=',')
            np.savetxt('respq_1x.csv',bandvec_q,delimiter=',')


