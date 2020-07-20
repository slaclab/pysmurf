#Runlike this exec(open("scratch/shawn/tune_1000x_noTES.py").read())

band=1

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

if band==0:
    #####################################
    # band 0
    S.set_att_uc(band,26)
    S.set_att_dc(band,18)
    resonly=[4095.08,4324.68,4345.73,4221.15,4287.94,4183.74,4222.49,4288.72,4443.77,4223.43,4291.01,4310.50,4224.67,4434.94,4167.34,4254.75,4436.38,4321.67,4255.37,4198.46,4323.14,4256.59]
    S.freq_resp=S.fake_resonance_dict(resonly)
    #####################################    
    
S.setup_notches(band,new_main_assignment=True)
S.plot_tune_summary(band,eta_scan=True)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

fraction_full_scale=0.44677
S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=None,
                 fraction_full_scale=fraction_full_scale,
                 make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,
                 feedback_start_frac=1/5.,feedback_end_frac=.98,meas_lms_freq=True)

