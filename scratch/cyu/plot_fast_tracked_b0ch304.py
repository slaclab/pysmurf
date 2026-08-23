plt.rc('font',family='serif')
plt.rc('font',size=16)


band = 0
ch = 304

# calculate the frequency offset
subband_centers = S.get_subband_centers(band)
sb_ctr = subband_centers[0][280]
band_center = S.get_band_center_mhz(band)
#freq_offset = S.get_center_frequency_mhz_channel(band,channel)
offset_mhz = band_center + sb_ctr
offset_khz = offset_mhz * 1.e3


# setup plots
fig,ax = plt.subplots(2,sharex=True,figsize=(12,8))


ax[0].plot(timevec[500:600],f[500:600] + offset_khz)
ax[0].set_ylabel('tracked frequency [kHz]')


# generate a sawtooth
st = (signal.sawtooth(2*np.pi*110.e3*timevec) + 1) * 2

ax[1].plot(timevec[500:600],st[500:600])
ax[1].set_ylabel('flux ramp [n$\Phi_0$]')
ax[1].set_xlabel('time [ms]')
ax[0].title('Tracked Frequency, Fast Flux Ramp')

plt.savefig(f'scratch/cyu/ltd_figs/fastfluxramp_b{band}ch{ch}.png',bbox_inches='tight')
ax.clf()
