# Runlike this exec(open("estimate_phase_delay.py").read())
# that way it will use your pysmurf S object.

import pysmurf
import numpy as np
import time
import sys

load_full_band_resp=True
load_find_freq=True
load_find_freq_check=True

band=2
n_samples=2**19
make_plot=True
save_data=True
n_scan=5

freq_min=-2.5E8
freq_max=2.5E8

# how to get AMC s/n and fw
def get_amcc_dump_bsi(ip='shm-smrf-sp01',slot=4):
    import subprocess
    result=subprocess.check_output(['amcc_dump_bsi','--all','%s/%d'%(ip,slot)])
    result_string=result.decode('utf-8')
    return result_string

S.band_off(band)
S.flux_ramp_off()

freq_cable=None
resp_cable=None
if load_full_band_resp:
    S.log('Loading full band resp data')
    fbr_path='/data/smurf_data/20190702/1562052474/outputs'
    fbr_ctime=1562052477
    fbr_freq_file=os.path.join(fbr_path,'%d_freq_full_band_resp.txt'%fbr_ctime)
    fbr_real_resp_file=os.path.join(fbr_path,'%d_real_full_band_resp.txt'%fbr_ctime)
    fbr_complex_resp_file=os.path.join(fbr_path,'%d_imag_full_band_resp.txt'%fbr_ctime)
    
    freq_cable = np.loadtxt(fbr_freq_file)
    real_resp_cable = np.loadtxt(fbr_real_resp_file)
    complex_resp_cable = np.loadtxt(fbr_complex_resp_file)
    resp_cable = real_resp_cable + 1j*complex_resp_cable
else:
    S.log('Running full band resp')
    freq_cable, resp_cable = S.full_band_resp(band, n_samples=n_samples, \
                                              make_plot=make_plot, \
                                              save_data=save_data, \
                                              n_scan=n_scan)
    
idx = np.where( (freq_cable > freq_min) & (freq_cable < freq_max) )

cable_z = np.polyfit(freq_cable[idx], np.unwrap(np.angle(resp_cable[idx])), 1)
cable_p = np.poly1d(cable_z)
cable_delay_us=np.abs(1.e6*cable_z[0]/2/np.pi)

freq_cable_subset=freq_cable[idx]
resp_cable_subset=resp_cable[idx]
#### done measuring cable delay

#### start measuring dsp delay (cable+processing)
# Zero refPhaseDelay and refPhaseDelayFine to get uncorrected phase
# delay.
# max is 7
S.set_ref_phase_delay(band,0)
# max is 255
S.set_ref_phase_delay_fine(band,0)

dsp_subbands=range(13,115)
freq_dsp=None
resp_dsp=None
if load_find_freq:
    S.log('Loading DSP frequency sweep data')
    ff_path='/data/smurf_data/20190702/1562052474/outputs'
    ff_ctime=1562052881
    ff_freq_file=os.path.join(ff_path,'%d_amp_sweep_freq.txt'%ff_ctime)
    ff_resp_file=os.path.join(ff_path,'%d_amp_sweep_resp.txt'%ff_ctime)

    freq_dsp=np.loadtxt(ff_freq_file)
    resp_dsp=np.loadtxt(ff_resp_file,dtype='complex')
else:
    S.log('Running find_freq')
    freq_dsp,resp_dsp=S.find_freq(band,dsp_subbands)
    ## not really faster if reduce n_step or n_read...somehow.
    #freq_dsp,resp_dsp=S.full_band_ampl_sweep(band, subband=dsp_subbands, drive=drive, n_read=2, n_step=n_step)

# only preserve data in the subband half width
n_subbands = S.get_number_sub_bands(band)
digitizer_frequency_mhz = S.get_digitizer_frequency_mhz(band)
subband_half_width = digitizer_frequency_mhz/\
                     n_subbands

subbands,subband_centers=S.get_subband_centers(band)
subband_freq_min=-subband_half_width/2.
subband_freq_max=subband_half_width/2.
freq_dsp_subset=[]
resp_dsp_subset=[]
for sb,sbc in zip(subbands,subband_centers):
    freq_subband=freq_dsp[sb]-sbc
    idx = np.where( ( freq_subband > subband_freq_min ) & (freq_subband < subband_freq_max) )
    freq_dsp_subset.extend(freq_dsp[sb][idx])
    resp_dsp_subset.extend(resp_dsp[sb][idx])

freq_dsp_subset=np.array(freq_dsp_subset)
resp_dsp_subset=np.array(resp_dsp_subset)

# to Hz
freq_dsp_subset=(freq_dsp_subset)*1.0E6

# fit
dsp_z = np.polyfit(freq_dsp_subset, np.unwrap(np.angle(resp_dsp_subset)), 1)
dsp_p = np.poly1d(dsp_z)
dsp_delay_us=np.abs(1.e6*dsp_z[0]/2/np.pi)

## compute refPhaseDelay and refPhaseDelayFine
digitizer_frequency_mhz=S.get_digitizer_frequency_mhz(band)
n_subband = S.get_number_sub_bands(band)
subband_half_width_mhz = digitizer_frequency_mhz/\
                     n_subband/2
refPhaseDelay=int(np.ceil(dsp_delay_us*subband_half_width_mhz))
refPhaseDelayFine=int(np.round((digitizer_frequency_mhz/2/subband_half_width_mhz)*(refPhaseDelay-dsp_delay_us*subband_half_width_mhz)))
processing_delay_us=dsp_delay_us-cable_delay_us

print('-------------------------------------------------------')
print('Estimated refPhaseDelay={}'.format(refPhaseDelay))
print('Estimated refPhaseDelayFine={}'.format(refPhaseDelayFine))
print('Estimated processing_delay_us={}'.format(processing_delay_us))
print('-------------------------------------------------------')

#### done measuring dsp delay (cable+processing)

#### start measuring total (DSP) delay with estimated correction applied
# Zero refPhaseDelay and refPhaseDelayFine to get uncorrected phase
# delay.
# max is 7
S.set_ref_phase_delay(band,refPhaseDelay)
# max is 255
S.set_ref_phase_delay_fine(band,refPhaseDelayFine)

freq_dsp_corr=None
resp_dsp_corr=None
if load_find_freq_check:
    S.log('Loading delay-corrected DSP frequency sweep data')
    ff_corr_path='/data/smurf_data/20190702/1562052474/outputs'
    ff_corr_ctime=1562053274
    ff_corr_freq_file=os.path.join(ff_corr_path,'%d_amp_sweep_freq.txt'%ff_corr_ctime)
    ff_corr_resp_file=os.path.join(ff_corr_path,'%d_amp_sweep_resp.txt'%ff_corr_ctime)
    
    freq_dsp_corr=np.loadtxt(ff_corr_freq_file)
    resp_dsp_corr=np.loadtxt(ff_corr_resp_file,dtype='complex')
else:
    S.log('Running find_freq')
    freq_corr_dsp,resp_corr_dsp=S.find_freq(band,dsp_subbands)

freq_dsp_corr_subset=[]
resp_dsp_corr_subset=[]
for sb,sbc in zip(subbands,subband_centers):
    freq_subband=freq_dsp_corr[sb]-sbc
    idx = np.where( ( freq_subband > subband_freq_min ) & (freq_subband < subband_freq_max) )
    freq_dsp_corr_subset.extend(freq_dsp_corr[sb][idx])
    resp_dsp_corr_subset.extend(resp_dsp_corr[sb][idx])

freq_dsp_corr_subset=np.array(freq_dsp_corr_subset)
resp_dsp_corr_subset=np.array(resp_dsp_corr_subset)

# to Hz
freq_dsp_corr_subset=(freq_dsp_corr_subset)*1.0E6

# fit
dsp_corr_z = np.polyfit(freq_dsp_corr_subset, np.unwrap(np.angle(resp_dsp_corr_subset)), 1)
dsp_corr_p = np.poly1d(dsp_corr_z)
dsp_corr_delay_us=np.abs(1.e6*dsp_corr_z[0]/2/np.pi)
#### done measuring total (DSP) delay with estimated correction applied

# plot unwraped phase in top panel, subtracted in bottom

fig, ax = plt.subplots(3, figsize=(6,7.5), sharex=True)

f_cable_plot = (freq_cable_subset) / 1.0E6
cable_phase = np.unwrap(np.angle(resp_cable_subset))

f_dsp_plot = (freq_dsp_subset) / 1.0E6
dsp_phase = np.unwrap(np.angle(resp_dsp_subset))

f_dsp_corr_plot = (freq_dsp_corr_subset) / 1.0E6
dsp_corr_phase = np.unwrap(np.angle(resp_dsp_corr_subset))

ax[0].set_title('Band {} Cable Delay'.format(band))
ax[0].plot(f_cable_plot,cable_phase,label='Cable (full_band_resp)',c='g',lw=3)
ax[0].plot(f_cable_plot,cable_p(f_cable_plot*1.0E6),'m--',label='Cable delay fit',lw=3)

ax[1].set_title('Band {} DSP Delay'.format(band))
ax[1].plot(f_dsp_plot,dsp_phase,label='DSP (find_freq)',c='c',lw=3)
ax[1].plot(f_dsp_plot,dsp_p(f_dsp_plot*1.0E6),c='orange',ls='--',label='DSP delay fit',lw=3)

ax[0].set_ylabel("Phase [rad]")
ax[0].set_xlabel('Frequency offset from band center [MHz]')

ax[1].set_ylabel("Phase [rad]")
ax[1].set_xlabel('Frequency offset from band center [MHz]')

ax[0].legend(loc='lower left')
ax[1].legend(loc='lower left')

bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)        
ax[0].text(.97, .90, 'cable delay={:.5f} us'.format(cable_delay_us),
           transform=ax[0].transAxes, fontsize=10,
           bbox=bbox,horizontalalignment='right')

ax[1].text(.97, .90, 'dsp delay={:.5f} us'.format(dsp_delay_us),
           transform=ax[1].transAxes, fontsize=10,
           bbox=bbox,horizontalalignment='right')

cable_residuals=cable_phase-(cable_p(f_cable_plot*1.0E6))
ax[2].plot(f_cable_plot,cable_residuals-np.median(cable_residuals),label='Cable (full_band_resp)',c='g')
dsp_residuals=dsp_phase-(dsp_p(f_dsp_plot*1.0E6))
ax[2].plot(f_dsp_plot,dsp_residuals-np.median(dsp_residuals),label='DSP (find_freq)',c='c')
ax[2].plot(f_dsp_corr_plot,dsp_corr_phase-np.median(dsp_corr_phase),label='DSP corrected (find_freq)',c='m')
ax[2].set_title('Band {} Residuals'.format(band))
ax[2].set_ylabel("Residual [rad]")
ax[2].set_xlabel('Frequency offset from band center [MHz]')
ax[2].set_ylim([-5,5])

ax[2].text(.97, .92, 'refPhaseDelay={}'.format(refPhaseDelay),
           transform=ax[2].transAxes, fontsize=8,
           bbox=bbox,horizontalalignment='right')
ax[2].text(.97, .84, 'refPhaseDelayFine={}'.format(refPhaseDelayFine),
           transform=ax[2].transAxes, fontsize=8,
           bbox=bbox,horizontalalignment='right')
ax[2].text(.97, .76, 'processing delay={:.5f} us'.format(processing_delay_us),
           transform=ax[2].transAxes, fontsize=8,
           bbox=bbox,horizontalalignment='right')
ax[2].text(.97, .68, 'delay post-correction={:.3f} ns'.format(dsp_corr_delay_us*1000.),
           transform=ax[2].transAxes, fontsize=8,
           bbox=bbox,horizontalalignment='right')

ax[2].legend(loc='upper left')

plt.tight_layout()
plt.show()

