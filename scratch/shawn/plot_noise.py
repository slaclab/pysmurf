import pysmurf
import numpy as np
from scipy import signal

grid_on=True
band=2
ch=385
detrend='constant'
nperseg=2**18
fs=200
low_freq=np.array([.1, 1.])
high_freq=np.array([1., 10.])
#why do I have to do this
S.config.get('smurf_to_mce')['filter_freq']=200
gcp_mode=True

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp28_smurfsrv04.cfg')


timestamp, phase, mask = S.read_stream_data_gcp_save('/data/smurf_data/20181216/1544999020/outputs/1545008527.dat_1545008529.dat')

phase *= S.pA_per_phi0/(2.*np.pi)

ch_idx = mask[band, ch]
f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg, 
                      fs=fs, detrend=detrend)
Pxx = np.sqrt(Pxx)

good_fit = False
try:
    popt,pcov,f_fit,Pxx_fit = S.analyze_psd(f,Pxx)
    wl,n,f_knee = popt
    if f_knee != 0.:
        wl_list.append(wl)
        f_knee_list.append(f_knee)
        n_list.append(n)
        good_fit = True    
    S.log('Band %i, ch. %i:' % (band,ch) + ' white-noise level = {:.2f}'.format(wl) +
             ' pA/rtHz, n = {:.2f}'.format(n) + 
             ', f_knee = {:.2f} Hz'.format(f_knee))
except:
    S.log('Band %i, ch. %i: bad fit to noise model' % (band,ch))

## make plot
plt.ion()
fig, ax = plt.subplots(2, figsize=(6,8))

sampleNums = np.arange(len(phase[ch_idx]))
t_array = sampleNums/fs

ax[0].plot(t_array,phase[ch_idx] - np.mean(phase[ch_idx]))
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Phase [pA]')
if grid_on:
    ax[0].grid()
    
ax[1].plot(f, Pxx)
if good_fit:
    ax[1].plot(f_fit,Pxx_fit,linestyle = '--',label=r'$n=%.2f$' % (n))
    ax[1].plot(f_knee,2.*wl,linestyle = 'none',marker = 'o',label=r'$f_\mathrm{knee} = %.1f\,\mathrm{mHz}$' % (f_knee*1000))
    ax[1].plot(f_fit,wl + np.zeros(len(f_fit)),linestyle = ':',label=r'$\mathrm{wl} = %.0f\,\mathrm{pA}/\sqrt{\mathrm{Hz}}$' % (wl))
    ax[1].legend(loc='best')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_xlim(f[1],f[-1])
ax[1].set_ylabel('Eq. TES Current Noise [pA/rtHz]')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
if grid_on:
    ax[1].grid()

res_freq = S.channel_to_freq(band, ch)
ax[0].set_title('Band {} Ch {:03} - {:.1f} MHz'.format(band, ch, res_freq))
plt.tight_layout()
