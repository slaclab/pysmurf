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

import numpy as np
import matplotlib.pylab as plt
 
of=open('%s_2pt4.dat'%S.get_timestamp(),'w+')
fmt='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}\n'
columns=['ctime','fres','filename','IorQ']
hdr=fmt.format(columns)
 
of.write(hdr)
of.flush()
print(hdr.rstrip())
 
ffs=[]
dfs=[]
 
nsamp=2**25
# 2 for 2.4Mhz, 1 for 600khz
single_channel_readout=2
 
# go to fixed tone.
S.set_feedback_enable(band,0)
# make sure we're on resonance.
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

# make sure flux ramp is off
S.flux_ramp_off()

def etaPhaseModDegree(etaPhase):
    return (etaPhase+180)%360-180
qEtaPhaseDegree=S.get_eta_phase_degree_channel(band,channel)
for IorQ in ['Q0','Q+','I+','I-']:
    if IorQ is 'Q0':        
        S.set_eta_phase_degree_channel(band,channel,qEtaPhaseDegree)
    if IorQ is 'Q+':
        S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree+180))                        
    if IorQ is 'I+':
        S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree+90))
    if IorQ is 'I-':
        S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree-90))                
    ctime1=int(S.get_timestamp())
    filename='%d.dat'%ctime1
    # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_sec=60.
    f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp, filename=str(ctime1));
    dfs.append(df)
    data=fmt.format([str(ctime1),'%0.6f'%(S.channel_to_freq(band,channel)),filename,IorQ])
    of.write(data)    
    of.flush()
S.set_eta_phase_degree_channel(band,channel,qEtaPhaseDegree)    
 
of.close()
