#Runlike this exec(open("scratch/jake/tune_1000x_noTES_nist.py").read())

band=2
if band==0:
    #####################################
    # band 0
    S.set_att_uc(band,26)
    S.set_att_dc(band,18)
    resonly=[4095.08,4324.68,4345.73,4221.15,4287.94,4183.74,4222.49,4288.72,4443.77,4223.43,4291.01,4310.50,4224.67,4434.94,4167.34,4254.75,4436.38,4321.67,4255.37,4198.46,4323.14,4256.59]
    S.freq_resp=S.fake_resonance_dict(resonly)
    #####################################
    
if band==1:
    #####################################
    # band 1
    #S.set_att_uc(band,20)
    S.set_att_uc(band,20)
    #S.set_att_dc(band,25)
    S.set_att_dc(band,0)
    resonly=[4882.38,4857.36,4612.17]
    S.freq_resp=S.fake_resonance_dict(resonly)
    #####################################

if band==2:
    #####################################
    # band 2
    S.set_att_uc(band,12)
    S.set_att_dc(band,31)
    resonly=[5095.34,5478.41,5421.14,5459.97,5487.78,5450.29,5065.48,5430.14,5201.24,5422.73,5461.34,5488.87,5431.47,5424.97,5462.78,5490.17,5453.73,5433.13,5492.00,5483.97,5445.02,5214.33,5426.40,5463.86,5417.21,5493.59,5454.89,5070.39,5434.92,5477.13,5024.25,5485.28,5446.57,5215.85,5427.69,5464.92,5419.56,5494.77,5455.94,5436.81,5130.06,5486.46,5488.05,5429.03,5033.78,5496.06,5456.81,5438.13,5458.27,5438.65]
    S.freq_resp=S.fake_resonance_dict(resonly)
    #####################################
    
S.setup_notches(band,new_master_assignment=True)
S.plot_tune_summary(band,eta_scan=True)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

fraction_full_scale=0.43894
S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=None,
                 fraction_full_scale=fraction_full_scale,
                 make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,
                 feedback_start_frac=1/5.,feedback_end_frac=.98,meas_lms_freq=True)

