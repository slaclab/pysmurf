# Runlike this exec(open("scratch/shawn/full_band_response.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
plt.ion()

bands=[2] #S.config.get('init').get('bands')

niter=1
n_scan=5
fit_freq_min=-150e6
fit_freq_max=150e6

results_dict={}

timestamp=S.get_timestamp()

results_dict['timestamp']=timestamp
results_dict['n_scan']=n_scan
results_dict['fit_freq_min']=fit_freq_min
results_dict['fit_freq_max']=fit_freq_max

for idx in range(niter):
    results_dict[band]={}    
    for band in bands:
        results_dict[band][idx]={}            
        print(' ')
        print(' ')
        print(f'Band {band}')
        print(' ')
        print(' ')
        
        freq, resp = S.full_band_resp(band=band,
                                      n_scan=n_scan,
                                      timestamp=timestamp)
        
        results_dict[band][idx]['freq']=freq
        results_dict[band][idx]['resp']=resp
        
        rf_phase = (
            np.unwrap(np.arctan2(np.imag(resp),np.real(resp))))
        
        # fit line to central frequency interval
        fit_idx = np.where( (freq > fit_freq_min) & (freq < fit_freq_max) )
        fit_z = np.polyfit( freq[fit_idx]/1.e6, rf_phase[fit_idx], 1)
        results_dict[band][idx]['fit'] = np.poly1d(fit_z)
        
nrows=1
if len(bands)>4:
    nrows=2
fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(14,7), sharex=True)

fig.suptitle(f'RF deterministic latency {timestamp}')
for band in bands:
    this_ax=None
    this_title=f'band {band}'
    if len(bands)>4:
        this_ax=ax[int(band/4)][band%4]
    else:
        this_ax=ax[band%4]
    this_ax.set_title(this_title)

    print('here')
    for idx in range(niter):        
        f_plot=results_dict[band][idx]['freq']
        resp_plot=results_dict[band][idx]['resp']
        rf_phase_plot = (
            np.unwrap(np.arctan2(np.imag(resp_plot),np.real(resp_plot))))        

        #this_ax.plot(f_plot/1.e6, rf_phase_plot)
        #this_ax.plot(f_plot/1.e6,results_dict[band][0]['fit'](f_plot/1.e6),'r--')

        this_ax.plot(f_plot/1.e6,(
            rf_phase_plot - results_dict[band][0]['fit'](f_plot/1.e6)),
            'r--')               
        #this_ax.plot(f_plot/1.e6, rf_phase_plot)
        #plt.xlim(fit_freq_min/1.e6,fit_freq_max/1.e6)

sys.exit(1)

save_name=f'{timestamp}_rfdetlat.npy'
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
