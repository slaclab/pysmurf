# Runlike this exec(open("scratch/shawn/full_band_response.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
from scipy import signal

n_scan_per_band=5
wait_btw_bands_sec=5

timestamp=S.get_timestamp()
bands=S.config.get('init').get('bands')
digitizer_frequency_mhz = S.get_digitizer_frequency_mhz()
data_length=2**19

resp_dict={}
for band in bands:
    print(' ')
    print(' ')
    print(f'Band {band}')
    print(' ')
    print(' ')
    resp_dict[band]={}
    resp_dict[band]['fc']=S.get_band_center_mhz(band)
    
    dac_dat=S.read_dac_data(band, do_plot=False,show_plot=False,save_plot=False,save_data=True,data_length=data_length);
    f_dac, p_dac = signal.welch(dac_dat, fs=digitizer_frequency_mhz, nperseg=data_length/2, return_onesided=False,detrend=False)
    resp_dict[band]['f_dac']=f_dac
    resp_dict[band]['p_dac']=p_dac

    adc_dat=S.read_adc_data(band, do_plot=False,show_plot=False,save_plot=False,save_data=True,data_length=data_length);
    f_adc, p_adc = signal.welch(adc_dat, fs=digitizer_frequency_mhz, nperseg=data_length/2, return_onesided=False,detrend=False) 
    resp_dict[band]['f_adc']=f_adc
    resp_dict[band]['p_adc']=p_adc
    
    time.sleep(wait_btw_bands_sec)

fig, axes = plt.subplots(2,2,figsize=(14,6), sharex=True)

color_cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
#ax[0].set_title(f'Full band response {timestamp}')
for band in bands:
    bay=1
    if band in range(4):
        bay=0

    dac_f_plot=resp_dict[band]['f_dac']
    dac_resp_plot=resp_dict[band]['p_dac']
    dac_plot_idx = np.where(np.logical_and(dac_f_plot>-250, dac_f_plot<250))[0]
    axes[0,bay].plot(dac_f_plot[dac_plot_idx]+resp_dict[band]['fc'], np.log10(np.abs(dac_resp_plot[dac_plot_idx])),label=f'b{band}',color=color_cycle[band])
       
    adc_f_plot=resp_dict[band]['f_adc']
    adc_resp_plot=resp_dict[band]['p_adc']
    adc_plot_idx = np.where(np.logical_and(adc_f_plot>-250, adc_f_plot<250))[0]
    axes[1,bay].plot(adc_f_plot[adc_plot_idx]+resp_dict[band]['fc'], np.log10(np.abs(adc_resp_plot[adc_plot_idx])),label=f'b{band}',color=color_cycle[band])

for bay in range(2):
    axes[0,bay].legend(loc='upper left',fontsize=8)    
    axes[1,bay].set_xlabel('Frequency [MHz]')

    axes[0,bay].set_ylabel("log10(abs(DAC))")
    axes[1,bay].set_ylabel("log10(abs(ADC))")        

    axes[0,bay].set_title(f'Bay {bay} ADC/DAC data ({timestamp})')
    
plt.tight_layout()
save_name = '{}_full_band_resp_all.png'.format(timestamp)
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name), 
            bbox_inches='tight')
plt.show()

# log plot file
#logf=open('scratch/shawn/scripts/loop_full_band_resps.txt','a+')
#logf.write(f'{os.path.join(S.plot_dir, save_name)}'+'\n')
#logf.close()


