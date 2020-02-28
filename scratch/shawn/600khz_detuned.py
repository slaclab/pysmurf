# If using the pole_upgrade branch of pysmurf, must change pysmurf so that you get this when
# you git diff
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#[cryo@smurf-srv07 pysmurf]$ git diff -r HEAD
#diff --git a/pysmurf/util/smurf_util.py b/pysmurf/util/smurf_util.py
#index 4811f7f..ef4ede1 100644
#--- a/pysmurf/util/smurf_util.py
#+++ b/pysmurf/util/smurf_util.py
#@@ -114,7 +114,7 @@ class SmurfUtilMixin(SmurfBase):
#         while not done:
#             done=True
#             for k in range(2):
#-                wr_addr = self.get_waveform_wr_addr(bay, engine=0)
#+                #wr_addr = self.get_waveform_wr_addr(bay, engine=0)
#                 empty = self.get_waveform_empty(bay, engine=k)
#                 if not empty:
#                     done=False
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# the script will run a lot faster if you turn off all other channels.
band=2
channel=0
dfsteps_khz=np.array([ -50., -40, -30, -20, -10, 0., 10., 20., 30., 40., 50.])

import numpy as np
import matplotlib.pylab as plt
 
of=open('%s_2pt4dt.dat'%S.get_timestamp(),'w+')
fmt='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}\n'
columns=['ctime','fres','filename','df']
hdr=fmt.format(columns)
 
of.write(hdr)
of.flush()
print(hdr.rstrip())
 
ffs=[]
dfs=[]
 
nsamp=2**25
#600 kHz
S.set_decimation(band,0)
# 32768 ends up negative when you read it back for some reason
S.set_filter_alpha(band,32767) # DDS filter.  f3dB ~ 275kHz
nsamp=2**25
# 2 for 2.4Mhz, 1 for 600khz
single_channel_readout=1

# go to fixed tone.
S.set_feedback_enable(band,0)

S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

fsb0=S.get_center_frequency_mhz_channel(band,channel)
for df_khz in dfsteps_khz:
    # eta scan at detune
    S.set_center_frequency_mhz_channel(band,channel,fsb0+df_khz/1000.)    
    S.run_serial_eta_scan(band)    
    ctime1=int(S.get_timestamp())
    filename='%d.dat'%ctime1
    # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.
    f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp, filename=str(ctime1));
    dfs.append(df)
    data=fmt.format([str(ctime1),'%0.6f'%(S.channel_to_freq(band,channel)),filename,df_khz])
    of.write(data)    
    of.flush()

# return the tone to centered
S.set_center_frequency_mhz_channel(band,channel,fsb0+df_khz/1000.)        
 
of.close()
