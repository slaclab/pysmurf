# from check_psds_b0.py have decided that ch304 is our guy
# making this script so the data-taking, plotting etc. are more repeatable

band = 0;
chs = [304];

for ch in chs:
    S.band_off(band)
    subband = S.get_subband_from_channel(band,ch)
    S.find_freq(band,subband=[subband],make_plot=True,show_plot=False)
    S.setup_notches(band,tone_power=13,new_master_assignment=True)
    S.run_serial_gradient_descent(band); S.run_serial_eta_scan(band)
    S.set_feedback_enable_channel(band,ch,0)

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
    plt.savefig(f'scratch/cyu/ltd_figs/iq_psd_nice_b0ch{ch}.png',bbox_inches='tight')
    plt.clf()
