for band in range(8):
    S.set_att_uc(band,0)
    S.set_att_dc(band,0)    
    sys.argv[1]=band; exec(open('scratch/shawn/fill_band.py').read())

for band in range(8):
    S.check_dac_saturation(band)
    S.check_adc_saturation(band)


for band in range(8):
    adc_data = S.read_adc_data(data_length=2**18, band=band, make_plot=False, show_plot=False)


