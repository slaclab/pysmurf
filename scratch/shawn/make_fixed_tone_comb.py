def fixed_tone_comb(S,band,Nfixed=100,keepout_mhz=2,fmin_mhz=-248,fmax_mhz=248,std_khz=250,drive=7):
    band_center_mhz=S.get_band_center_mhz(band)
    fmin_mhz=fmin_mhz+band_center_mhz
    fmax_mhz=fmax_mhz+band_center_mhz
    tracking_freqs=[S.channel_to_freq(band,chan) for chan in S.which_on(band)]
    std_mhz=std_khz/1000.
    fixed_tone_freqs=np.linspace(fmin_mhz,fmax_mhz,Nfixed)+(np.random.rand(Nfixed)-0.5)*std_mhz
    for ftf in fixed_tone_freqs:
    	if np.min(np.abs(ftf-tracking_freqs))>keepout_mhz:
	   S.set_fixed_tone(ftf,drive)

fixed_tone_comb(S,3,Nfixed=256,std_khz=750.,drive=7)
