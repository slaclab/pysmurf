# the script will run a lot faster if you turn off all other channels.
bg=7 # TES bias group
band=2
channel=0
 
# Shawn measured a phi0 to be 0.253 in the units that the S.set_fixed_flux_ramp_bias
# function takes at Pole in band 3.
ffphi0=0.253
# take this many points between ff=0 and ff=ffphi0
nffsteps=25
 
import numpy as np
import matplotlib.pylab as plt
import time

of=open('%s_fsnt.dat'%S.get_timestamp(),'w+')
fmt='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}\n'
columns=['ctime','ff','fres','filename']
hdr=fmt.format(columns)
 
of.write(hdr)
of.flush()
print(hdr.rstrip())
 
ffs=[]
dfs=[]
 
# settings for taking fast(ish) data
#S.set_decimation(band,5) # decimate by 2^5=32.  Sample rate in single_channel_readout=1 is 600kHz, so this will be 18750 Hz.
#S.set_filter_alpha(band,1638) # DDS filter.  I forget what f3dB this was, but it's about right for decimation=5.
 
#600 kHz
S.set_decimation(band,0)
# 32768 ends up negative when you read it back for some reason
S.set_filter_alpha(band,32767) # DDS filter.  f3dB ~ 275kHz
nsamp=2**25
# 2 for 2.4Mhz, 1 for 600khz
single_channel_readout=1

# maintain zero
bgzerovolts=S.get_tes_bias_bipolar(bg)

def play_waveform(S,bg,bgzerovolts):
    bgzero=bgzerovolts/S._rtm_slow_dac_bit_to_volt/2.
    multiplier=1/10000.
    scale = 2**17;
    sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/2048) + bgzero
    S.play_tes_bipolar_waveform(bg,sig)

def stop_waveform(S,bg,bgzerovolts):
    S.stop_tes_bipolar_waveform(bg)
    S.set_tes_bias_bipolar(bg,bgzerovolts)

# go to fixed tone.
S.set_feedback_enable(band,0)
# make sure we're on resonance.
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)
 
ffs=np.linspace(0,ffphi0,nffsteps)
for ffrb in ffs:
    S.set_fixed_flux_ramp_bias(ffrb)
    # want to stay on resonance.  flux ramp changed, so resonator frequency changed.  Re-center
    # tone on resonator's new position.
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)

    play_waveform(S,bg,bgzerovolts)
    time.sleep(5)    
    
    ctime1=int(S.get_timestamp())
    filename='%d.dat'%ctime1
    # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_sec=60.
    f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp, filename=str(ctime1));
    dfs.append(df)
    data=fmt.format([str(ctime1),'%0.6f'%(ffrb),'%0.6f'%(S.channel_to_freq(band,channel)),filename])
    of.write(data)
    of.flush()

    stop_waveform(S,bg,bgzerovolts)
    time.sleep(5)
 
S.unset_fixed_flux_ramp_bias(ffrb)
 
of.close()
