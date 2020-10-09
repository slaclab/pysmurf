# Start with all tones off.
# Set a tone at the frequency of the channel that always steps.  This should work:
#S.set_fixed_tone(5733.307683,12)
# See which channel it is with S.which_on(band).
#In [84]: S.which_on(3)
#Out[84]: array([287])
# and set band and channel appropriately, then
# run the script
band=3
channel=287

import numpy as np
import matplotlib.pylab as plt
 
of=open('%s_2pt4lb.dat'%S.get_timestamp(),'w+')
fmt='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}\n'
columns=['ctime','fres','filename','etaPhaseDegree']
hdr=fmt.format(columns)
 
of.write(hdr)
of.flush()
print(hdr.rstrip())
 
ffs=[]
dfs=[]
 
nsamp=2**25
# 2 for 2.4Mhz, 1 for 600khz
single_channel_readout=2
 
# make sure we're fixed tone.
S.set_feedback_enable(band,0)

# make sure etaMag=1
S.set_eta_mag_scaled_channel(band,channel,1)

for etaPhaseDegree in [0,-180,-90,90]:
    S.set_eta_phase_degree_channel(band,channel,etaPhaseDegree)
    ctime1=int(S.get_timestamp())
    filename='%d.dat'%ctime1
    # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_sec=60.
    f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp, filename=str(ctime1));
    dfs.append(df)
    data=fmt.format([str(ctime1),'%0.6f'%(S.channel_to_freq(band,channel)),filename,etaPhaseDegree])
    of.write(data)    
    of.flush()
 
of.close()
