#Runlike this exec(open("scratch/jake/tune_1000x_first_time_nist.py").read())
#this will generate a new tune file with the below attenuator and flux ramp settings

import numpy as np

bands=[0,1,2,3]

#turn everything off
#S.all_off()

#set up attenuator levels optimized for noise
S.set_att_uc(0,26)
S.set_att_uc(1,20)
S.set_att_uc(2,16)
S.set_att_uc(3,12)

S.set_att_dc(0,18)
S.set_att_dc(1,24)
S.set_att_dc(2,30)
S.set_att_dc(3,31)

#set amplitude scale
drive_power = 12 #do not go above 12

# for 2x inline 465 Ohm on both legs of split flux ramp (4x split for 1000x)
### 20kHz on bands 2&3
fraction_full_scale=0.43894
lms_freq_hz={}
lms_freq_hz[0]=19679
lms_freq_hz[1]=19679
lms_freq_hz[2]=19996
lms_freq_hz[3]=19996

# tune each band
for band in bands:
    #S.set_amplitude_scales(band,drive_power)
    S.find_freq(band,drive_power=drive_power,make_plot=True,show_plot=False,save_plot=True)
    S.setup_notches(band,drive=drive_power,new_main_assignment=True)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    # only plot first ten channels 
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=lms_freq_hz[band],
                     fraction_full_scale=fraction_full_scale,
                     make_plot=True,show_plot=False,channel=S.which_on(band)[0:10],nsamp=2**18,
                     feedback_start_frac=1/5.,feedback_end_frac=.98,meas_lms_freq=False)
