import scipy.signal as signal

band=0
ch=211

# now take data
q,i,sync = S.take_debug_data(band=band,channel=ch,rf_iq=True,nsamp=2**24)
eta_phase_rad = np.deg2rad(S.get_eta_phase_degree_channel(band,ch))
I = np.cos(eta_phase_rad)*i - np.sin(eta_phase_rad)*q
Q = np.sin(eta_phase_rad)*i + np.cos(eta_phase_rad)*q

ffi,pxxi = signal.welch(I,fs=2.4e6,nperseg=2**14)
ffq,pxxq = signal.welch(Q,fs=2.4e6,nperseg=2**14)

magfac = np.mean(q)**2 + np.mean(i)**2
pxxi_dbc = 10.*np.log10(pxxi/magfac)
pxxq_dbc = 10.*np.log10(pxxq/magfac)

plt.rc('font',family='serif')
plt.rc('font',size=12)
plt.figure(figsize=(4,4))
plt.semilogx(ffi,pxxi_dbc,alpha=0.8,label='I')
plt.semilogx(ffq,pxxq_dbc,alpha=0.8,label='Q')
plt.title(f'I/Q noise of fixed tone readout')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectral density [dBc/Hz]')
#plt.savefig(f'scratch/cyu/ltd_figs/iq_psd_nice_b0ch{ch}.png',bbox_inches='tight')
