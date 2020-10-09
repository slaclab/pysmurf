# Runlike this exec(open("scratch/shawn/full_band_response.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
plt.ion()

enable_amps=True
amplifier_wait_sec=15
if enable_amps:
    S.C.write_ps_en(11)
    time.sleep(amplifier_wait_sec)

timestamp=S.get_timestamp()
bands=S.config.get('init').get('bands')

subband=63
drive=12
Npts=500
n_read=100
f_sweep_half=S.get_channel_frequency_mhz()/2.
df_sweep=f_sweep_half/Npts
f_sweep=np.arange(-f_sweep_half,f_sweep_half,df_sweep)

results_dict={}

results_dict['timestamp']=timestamp
results_dict['Npts']=Npts
results_dict['n_read']=n_read
results_dict['drive']=drive
results_dict['subband']=subband
results_dict['f_sweep_half']=f_sweep_half
results_dict['f_sweep']=f_sweep
results_dict['df_sweep']=df_sweep

for band in bands:
    print(' ')
    print(' ')
    print(f'Band {band}')
    print(' ')
    print(' ')
    results_dict[band]={}

    subband_offset=S.get_subband_centers(band,as_offset=True)[1][subband]
    subband_center=S.get_subband_centers(band,as_offset=False)[1][subband]

    results_dict[band]['subband_offset']=subband_offset
    results_dict[band]['subband_center']=subband_center

    freq, resp = S.fast_eta_scan(band,
                                 subband,
                                 f_sweep+subband_offset,
                                 n_read,
                                 drive)
    
    results_dict[band]['freq']=freq
    results_dict[band]['resp']=resp

nrows=1
if len(bands)>4:
    nrows=2
fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(14,7), sharex=True)

fig.suptitle(f'Deterministic latency {timestamp}')
for band in bands:
    this_ax=None
    this_title=f'band {band}'
    if len(bands)>4:
        this_ax=ax[int(band/4)][band%4]
    else:
        this_ax=ax[band%4]
    this_ax.set_aspect('equal')
    this_ax.set_title(this_title)
    #f_plot=results_dict[band]['freq']/1e6
    resp_plot=results_dict[band]['resp']
    #ax[0].plot(f_plot[plot_idx]+results_dict[band]['fc'], np.log10(np.abs(resp_plot[plot_idx])),label=f'b{band}')
    this_ax.plot(np.real(resp_plot),np.imag(resp_plot))

save_name=f'{timestamp}_detlat.npy'
save_path=os.path.join(S.output_dir, save_name)
print(f'Saving data to {save_path}')

## Save data to disk for lookup
np.save(save_path, results_dict)

plt.tight_layout()

print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name.replace('.npy','png')),
            bbox_inches='tight')

## log plot file
logf=open('/data/smurf_data/smurf_loop.log','a+')
logf.write(f'{save_path}'+'\n')
logf.close()

print('Done running deterministic_latency.py.')
