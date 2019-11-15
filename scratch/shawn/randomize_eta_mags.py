for band in range(8):
    print(band)
    ema=S.get_eta_mag_array(band)    
    S.set_eta_mag_array(band,.5*np.ones_like(ema)+np.random.rand(len(ema))/10.)
    time.sleep(5)
    

