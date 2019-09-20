multiplier=1/100.
scale = 2**17;
sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/2048);

S.play_tes_bipolar_waveform(7,sig)
S.stop_tes_bipolar_waveform(7)
