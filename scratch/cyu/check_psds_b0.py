# dumb script to loop through channels and check the psds looking for one without the dumb bump
# will think more about where the bump comes from later

chs = [480,208,464,304,296,488,408,184,276,180,76,44,220];

for ch in chs:
    S.band_off(0)
    subband = S.get_subband_from_channel(0,ch)
    S.find_freq(0,subband=[subband],make_plot=True,show_plot=False)
    S.setup_notches(0,tone_power=13,new_master_assignment=True)
    S.run_serial_gradient_descent(0); S.run_serial_eta_scan(0)
    S.set_feedback_enable_channel(0,ch,0)

    # now take data
    q,i,sync = S.take_debug_data(band=0,channel=ch,rf_iq=True,nsamp=2**22)
    eta_phase_rad = np.deg2rad(S.get_eta_phase_degree_channel(0,ch))
    I = np.cos(eta_phase_rad)*i - np.sin(eta_phase_rad)*q
    Q = np.sin(eta_phase_rad)*i + np.cos(eta_phase_rad)*q

    ffi,pxxi = signal.welch(I,fs=2.4e6,nperseg=2**14)
    ffq,pxxq = signal.welch(Q,fs=2.4e6,nperseg=2**14)

    magfac = np.mean(q)**2 + np.mean(i)**2
    pxxi_dbc = 10.*np.log10(pxxi/magfac)
    pxxq_dbc = 10.*np.log10(pxxq/magfac)

    plt.figure()
    plt.semilogx(ffi,pxxi_dbc,alpha=0.8,label='I')
    plt.semilogx(ffq,pxxq_dbc,alpha=0.8,label='Q')
    plt.title(f'I/Q dBc/Hz noise b0ch{ch}')
    plt.legend()
    plt.savefig(f'scratch/cyu/iq_psd_b0ch{ch}.png')
    plt.clf()
