import numpy as np

bands=[0,1,2,3]

S.all_off()

S.set_att_uc(0,26)
S.set_att_uc(1,20)
S.set_att_uc(2,16)
S.set_att_uc(3,12)

S.set_att_dc(0,18)
S.set_att_dc(1,24)
S.set_att_dc(2,30)
S.set_att_dc(3,31)

# for 2x inline 465 Ohm on both legs of split flux ramp (4x split for 1000x)
### 20kHz on bands 2&3
#fraction_full_scale=0.4295
fraction_full_scale=0.43894
lms_freq_hz={}
lms_freq_hz[0]=19679
lms_freq_hz[1]=19679
lms_freq_hz[2]=19996
lms_freq_hz[3]=19996

# should load for 0,1,2,3 which we took at low power
S.load_tune('/data/smurf_data/tune/1582056629_tune.npy')

# relock bands
for band in bands:
    S.relock(band)

# kill known bad channels in this tune, mostly collisions, found with bad eta
b0_list = np.array([19, 82, 15])
b1_list = np.array([82,83,34,434,10,138,42,154,474,314])
b2_list = np.array([93,28,70,121,313,441,197,389,37,293,21,277,341,267,296,424,488,24,88,280,36])
b3_list = np.array([11,156,429,29])

bad_eta_list = {
    "0" : b0_list,
    "1" : b1_list,
    "2" : b2_list,
    "3" : b3_list
}

for b in np.sort(list(bad_eta_list.keys())):
    print (f'Band {b}')
    for ch in bad_eta_list[b]:
        S.channel_off(b, ch, write_log=True)        

#had bad tracking plots (some double flux ramped, some none)
b0_fr_list = np.array([285,335,420,442])
b1_fr_list = np.array([6,26,58,70,74,102,106,134,186,218,230,234,250,262,266,282,298,326,330,346,390,394,426,442,454,48,293])
b2_fr_list = np.array([8,53,10,51,69,72,149,181])
b3_fr_list = np.array([0,44,64,268,335,204,423,511])

bad_tracking_list = {
    "0" : b0_fr_list,
    "1" : b1_fr_list,
    "2" : b2_fr_list,
    "3" : b3_fr_list
}

for b in np.sort(list(bad_tracking_list.keys())):
    print (f'Band {b}')
    for ch in bad_tracking_list[b]:
        S.channel_off(b, ch, write_log=True)


# no point tuning before killing known bad channels
for band in bands:
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    # only plot first three channels 
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=lms_freq_hz[band],
                     fraction_full_scale=fraction_full_scale,
                     make_plot=True,show_plot=False,channel=S.which_on(band)[0:3],nsamp=2**18,
                     feedback_start_frac=1/5.,feedback_end_frac=.98,meas_lms_freq=False)
