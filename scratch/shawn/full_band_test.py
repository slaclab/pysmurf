# Runlike this exec(open("band_filling_test.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys

bands=S.config.get('init').get('bands')

amplitude=10 
wait=True 
wait_time_sec=0.1

# optional argument
timestamp=None
#if len(sys.argv)>1:
#    timestamp=int(sys.argv[1])

freqs = ((-2.4, -1.1, 0.7, 2.1))


channels=[] 

#for sb in np.arange(64,116):
for sb in np.arange(12,116):     
    for i in range(4):
        channels.append(S.get_channels_in_subband(bands[0],sb)[i]) 

j = 0
freq_array = np.zeros((1,512))
amp_array  = np.zeros((1,512), 'int')
for ch in channels: 
    freq_array[0][ch] = 0.1*np.random.rand(1)+freqs[j%4]
    amp_array[0][ch] = amplitude
    j = j + 1


for band in bands:
    print(' ')
    print(' ')
    print(band)
    print(' ')
    print(' ')
    S.set_center_frequency_array(band, freq_array[0] + 0.1*np.random.rand(1,512)[0])
    S.set_amplitude_scale_array(band, amp_array[0])
    S.set_eta_mag_array(band, np.ones((1,512))[0])

time.sleep(2)

# Save DAC and ADC spectra
for band in bands:
    S.read_dac_data(band, show_plot=False, do_plot=True, plot_ylimits=[1e-8,1e3],timestamp=timestamp)
    S.read_adc_data(band, show_plot=False, do_plot=False, plot_ylimits=[1e-8,1e3],timestamp=timestamp)

print('Done with half-band filling test.')
