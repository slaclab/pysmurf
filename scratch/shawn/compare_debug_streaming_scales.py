channel=79
num_debug_samples=2**20

# debug data
f_dd, dF_dd, sync_dd = S.take_debug_data(band=0, channel=channel, nsamp=num_debug_samples, IQstream=0, single_channel_readout=2)

t_dd = np.arange(len(f_dd))
t_dd = t_dd*1000/2.4e6

freq_dd, pxx_dd = signal.welch(f_dd, nperseg=len(f_dd)/8, fs=2.4e6)
pxx_db_dd=10*np.log10(pxx_dd)

# streaming
stream_datafn=S.take_stream_data(num_debug_samples/2.4e6) # matching time width of debug data
print(f'stream_datafn={stream_datafn}')
timestamp_s, freq_s, mask_s = S.read_stream_data(stream_datafn)

### compare

boxcar_length=240
coef       = 1/float(boxcar_length)
coef_fixed = np.round(coef*2**15)
filter_gain=coef_fixed*boxcar_length*2**-15

plt.figure()
plt.subplot(211)

asd_dd=1.e6*np.sqrt(10**(pxx_db_dd/10.))
plt.loglog(freq_dd, asd_dd,label=f'debug data ch{channel}')

# _s stands for streamed data
idx_s=mask_s[0][channel]
freq_s, pxx_s = signal.welch(freq_s[idx_s], nperseg=len(freq_s[idx_s])//2, fs=S.get_sample_frequency())
asd_s=np.sqrt(pxx_s)
#plt.plot(freq[0]/2**2*1.2e6/np.pi/filter_gain)
plt.loglog(freq_s,asd_s*1.2e6/np.pi/filter_gain/2/2/boxcar_length,label=f'stream channel {channel}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Hz/rt.Hz')
plt.legend(loc='lower left')

plt.subplot(211)

sb=S.get_subband_from_channel(0,channel)
freq_s_corr=freq[idx_s]*1.2e6/np.pi/filter_gain/2/2/boxcar_length
f_dd_hz=1e6*((f_dd)+S.get_band_center_mhz(0)+S.get_subband_centers(0)[1][sb])
plt.plot(t_dd,f_dd_hz,label=f'debug data chan{channel}')

f_s_hz=freq_s_corr+1.e6*(S.get_band_center_mhz(0)+S.get_center_frequency_mhz_channel(band=0, channel=channel)+S.get_subband_centers(0)[1][sb])
plt.plot((timestamp_s-timestamp_s[0])/1.e9,f_s_hz,label=f'stream data chan79')

plt.ylabel('Resonator frequency (Hz)')
plt.xlabel('Time (sec)')
plt.legend(loc='upper right')




