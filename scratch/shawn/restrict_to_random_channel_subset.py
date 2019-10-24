for band in range(8):
    restrict_nper_band=250
    print('-> Restricting nchan to %d.'%restrict_nper_band)
    import random
    assigned_channels=np.where(asa!=0)[0]
    ntotal=len(assigned_channels)
    n2kill=(ntotal-restrict_nper_band)
    channels2kill=random.sample(list(assigned_channels),n2kill)
    cfa[channels2kill]=0
    asa[channels2kill]=0
    
    S.set_center_frequency_array(band,cfa)
    S.set_amplitude_scale_array(band,asa)
