import matplotlib
matplotlib.use('Agg')

import pysmurf
import numpy as np
import time
import sys

## instead of takedebugdata try relaunch PyRogue, then loopFilterOutputArray, which is 
## the integral tracking term with lmsEnable[1..3]=0

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='smurf_server_s5',cfg_file='/home/cryo/docker/pysmurf/hb-devel-dspv2/pysmurf/cfg_files/experiment_fp29_smurfsrv03_noExtRef_hbOnlyBay0.cfg')

#######
hbInBay0=False
bands=[2,3]
Npts=3
bias=None
#wait_time=.05
wait_time=1.
#bias_low=-0.432
#bias_high=0.432
bias_low=-0.8
bias_high=0.8
Nsteps=250
#Nsteps=25
bias_step=np.abs(bias_high-bias_low)/float(Nsteps)
channels=None
#much slower than using loopFilterOutputArray,
#and creates a bunch of files
use_take_debug_data=False

# Look for good channels
if channels is None:
    channels = {}
    for band in bands:
        channels[band] = S.which_on(band)
print(channels[band])

if bias is None:
    bias = np.arange(bias_low, bias_high, bias_step)

# final output data dictionary
raw_data = {}
print(channels[band])
bands_with_channels_on=[]
for band in bands:
    print(band,channels[band])
    if len(channels[band])>0:
        S.log('{} channels on in band {}, configuring band for simple, integral tracking'.format(len(channels[band]),band))
        S.log('-> Setting lmsEnable[1-3] and lmsGain to 0 for band {}.'.format(band), S.LOG_USER)
        S.set_lms_enable1(band, 0)
        S.set_lms_enable2(band, 0)
        S.set_lms_enable3(band, 0)
        S.set_lms_gain(band, 0)

        raw_data[band]={}

        bands_with_channels_on.append(band)

bands=bands_with_channels_on

#amplitudes=[9,10,11,12,13,14,15]
# [None] means don't change the amplitude, but still retunes
amplitudes=[None]
for amplitude in amplitudes:

    ### begin retune on all bands with tones
    for band in bands:
        S.log('Retuning at tone amplitude {}'.format(amplitude))
        if amplitude is not None:
            S.set_amplitude_scale_array(band,np.array(S.get_amplitude_scale_array(band)*amplitude/np.max(S.get_amplitude_scale_array(band)),dtype=int))
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        raw_data[band][amplitude]={}
        
    ### end retune
    S.log('Starting to take flux ramp with amplitude={}.'.format(amplitude), S.LOG_USER)

    sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
    S.set_fixed_flux_ramp_bias(bias_low)
    time.sleep(wait_time)

    fs={}
    for band in bands:
        fs[band]=[]

    for b in bias:
        sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b)
        time.sleep(wait_time)

        for band in bands:
            if use_take_debug_data:
                f,df,sync=S.take_debug_data(band,IQstream=False,single_channel_readout=0)
                fsampmean=np.mean(f,axis=0)
                fs[band].append(fsampmean)
            else:
                fsamp=np.zeros(shape=(Npts,len(channels[band])))
                for i in range(Npts):
                    fsamp[i,:]=S.get_loop_filter_output_array(band)[channels[band]]
                fsampmean=np.mean(fsamp,axis=0)
                fs[band].append(fsampmean)

    sys.stdout.write('\n')

    S.log('Done taking flux ramp with amplitude={}.'.format(amplitude), S.LOG_USER)

    for band in bands:

        fres=[S.channel_to_freq(band, ch) for ch in channels[band]]
        raw_data[band][amplitude]['fres']=np.array(fres) + (2e3 if hbInBay0 else 0)
        raw_data[band][amplitude]['channels']=channels[band]

        if use_take_debug_data:
            #stack
            fovsfr=np.dstack(fs[band])[0]
            [sbs,sbc]=S.get_subband_centers(band)
            fvsfr=fovsfr[channels[band]]+[sbc[np.where(np.array(sbs)==S.get_subband_from_channel(band,ch))[0]]+S.get_band_center_mhz(band) for ch in channels[band]]
            raw_data[band][amplitude]['fvsfr']=fvsfr + (2e3 if hbInBay0 else 0)
        else:
            #stack
            lfovsfr=np.dstack(fs[band])[0]
            raw_data[band][amplitude]['lfovsfr']=lfovsfr
            raw_data[band][amplitude]['fvsfr']=np.array([arr/4.+fres for (arr,fres) in zip(lfovsfr,fres)]) + (2e3 if hbInBay0 else 0)

raw_data['bias'] = bias

# done - zero and unset
S.set_fixed_flux_ramp_bias(0)
S.unset_fixed_flux_ramp_bias()

import os
fn_raw_data = os.path.join('./', '%s_fr_sweep_data.npy'%(S.get_timestamp()))
np.save(fn_raw_data, raw_data)
