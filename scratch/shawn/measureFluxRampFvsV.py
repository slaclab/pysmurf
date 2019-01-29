import pysmurf
import numpy as np
import time
import sys

## instead of takedebugdata try relaunch PyRogue, then loopFilterOutputArray, which is 
## the integral tracking term with lmsEnable[1..3]=0

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp28_smurfsrv04.cfg')

#######
band=3
Npts=3
bias=None
wait_time=.05
bias_low=-0.432
bias_high=0.432
bias_step=.002
show_plot=False
make_plot=True
save_plot=True 
channels=None
gcp_mode=True
grid_on=False
#much slower than using loopFilterOutputArray,
#and creates a bunch of files
use_take_debug_data=False

# Look for good channels
if channels is None:
    channels = S.which_on(band)

if bias is None:
    bias = np.arange(bias_low, bias_high, bias_step)

S.set_lms_enable1(band, 0)
S.set_lms_enable2(band, 0)
S.set_lms_enable3(band, 0)

S.log('Turning lmsGain to 0.', S.LOG_USER)
lms_gain = S.get_lms_gain(band)
S.set_lms_gain(band, 0)

S.log('Staring to take flux ramp.', S.LOG_USER)

sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
S.set_fixed_flux_ramp_bias(bias_low)
time.sleep(wait_time)

fs=[]
for b in bias:
    sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
    sys.stdout.flush()
    S.set_fixed_flux_ramp_bias(b)
    time.sleep(wait_time)

    if use_take_debug_data:
        f,df,sync=S.take_debug_data(band,IQstream=False,single_channel_readout=0)
        fsampmean=np.mean(f,axis=0)
        fs.append(fsampmean)
    else:
        fsamp=np.zeros(shape=(Npts,len(channels)))
        for i in range(Npts):
            fsamp[i,:]=S.get_loop_filter_output_array(band)[channels]
        fsampmean=np.mean(fsamp,axis=0)
        fs.append(fsampmean)

sys.stdout.write('\n')

# done - zero and unset
S.set_fixed_flux_ramp_bias(0)
S.unset_fixed_flux_ramp_bias()

raw_data = {}

fres=[S.channel_to_freq(band, ch) for ch in channels]
raw_data['fres']=fres
raw_data['channels']=channels

if use_take_debug_data:
    #stack
    fovsfr=np.dstack(fs)[0]
    [sbs,sbc]=S.get_subband_centers(band)
    fvsfr=fovsfr[channels]+[sbc[np.where(np.array(sbs)==S.get_subband_from_channel(band,ch))[0]]+S.get_band_center_mhz(band) for ch in channels]
    raw_data['fvsfr']=fvsfr
else:
    #stack
    lfovsfr=np.dstack(fs)[0]
    raw_data['lfovsfr']=lfovsfr[channels]
    raw_data['fvsfr']=np.array([arr/4.+fres for (arr,fres) in zip(lfovsfr,fres)])

raw_data['bias'] = bias
raw_data['band'] = band

import os
fn_raw_data = os.path.join('./', '%s_fr_sweep_data.npy'%(S.get_timestamp()))
np.save(fn_raw_data, raw_data)

###
import matplotlib.pylab as plt
# if loading from file *.item(); e.g.;
#d=np.load('1543444743_fr_sweep_data.npy')
#raw_data=d.item()
def plot(chans,raw_data):
    for ch in chans:
        chf=raw_data['fvsfr'][np.where(raw_data['channels']==ch)[0][0]]
        chf=chf-np.mean(chf)
        plt.plot(raw_data['bias'],chf,label='ch%d'%ch)
    plt.xlabel('fraction fullscale (one-sided)')
    plt.ylabel('Foff (MHz)')
    plt.legend()
    plt.show()
###

sys.exit(1)

plot([385],raw_data)

## try to analyze
def fourier(x, *a):
    ret = a[3] + a[2] * np.sin(np.pi / a[1] * (x - a[0] ) )
    for deg in range(4, len(a)):
        ret += a[deg] * np.sin((deg+1) * np.pi / a[1] * (x - a[0]) )
    return ret

from scipy import signal
from scipy.optimize import curve_fit
def analyze(ch,raw_data):
    chf=raw_data['fvsfr'][np.where(raw_data['channels']==ch)[0][0]]
    bias=raw_data['bias']
    
    #peakind = signal.find_peaks_cwt(chf, np.linspace(0.05,0.4,4))

    biasspan=np.max(bias)-np.min(bias)
    fspan=np.max(chf)-np.min(chf)
    fmean=np.mean(chf)
    meanschf=chf-fmean
    peakind=signal.find_peaks(meanschf,threshold=0,distance=20)[0]

    Fres=S.channel_to_freq(raw_data['band'],ch)
    if len(peakind)==0:
        return
    try:
        qest=np.abs(bias[peakind[1]]-bias[peakind[0]])/2.
        phest=np.max(np.diff(bias[peakind]))
    except:
        plt.plot(bias,chf,label='ch%d data'%(ch))    
        plt.title('ch%d -- FAILED'%(ch))
        print('ch%d FAILED!'%ch)
        plt.legend()
        plt.show()
        return

    Fres=S.channel_to_freq(raw_data['band'],ch)
    print('%d\t%0.3f\t%0.4f\t%0.4f\n'%(ch,Fres,phest,fspan))
    of=open('phest.dat','a+')
    of.write('%d\t%0.3f\t%0.4f\n'%(ch,Fres,phest))
    of.close()

    plt.plot(bias,chf,label='ch%d data'%(ch))    
    plt.scatter(bias[peakind],chf[peakind],label='maxes')

    plt.title('ch%d  Fres=%0.3f  phest=%0.4f%%'%(ch,Fres,phest))

#    # fit first harmonic
#    p0=[qest+phest/2,qest,fspan/2,fmean]
#    param_bounds=([0,        biasspan/20., -np.inf,-np.inf],
#                  [2.*np.pi,     biasspan,  np.inf, np.inf])
#    popt, pcov = curve_fit(fourier, bias, chf, p0, bounds=param_bounds)
#
#    plt.plot(bias,fourier(bias,*p0),'g--',label='1st harmonic guess')
#    plt.plot(bias,fourier(bias,*popt),label='fit')
#    
    plt.legend()
    plt.savefig('ch%d.png'%ch)

#for ch in raw_data['channels']:                                    
#    analyze(ch,raw_data)                                              
#    plt.waitforbuttonpress()                                          
#    plt.clf() 
