# Runlike this exec(open("band_filling_test.py").read())
# to use the pysmurf S object you've already initialized
import time
import sys
import numpy as np
import random

band=int(sys.argv[1])

amplitude=10
freq_std_khz=100
freq_std_mhz=freq_std_khz/1000.
one_subband_at_a_time=False
wait_btw_subbands_sec=0.1

# if you want only N tones per band, set this
# to N.  Otherwise you'll get the max possible,
# if None.  Only used if one_subband_at_a_time is False
restrict_nper_band=None
#restrict_nper_band=231

print('filling bands %s'%str(band))

n_subbands = S.get_number_sub_bands(band)
digitizer_frequency_mhz = S.get_digitizer_frequency_mhz(band)
subband_half_width = digitizer_frequency_mhz / (n_subbands / 2) / 2
n_channels_per_subband=int(S.get_number_channels()/S.get_number_sub_bands())

tone_spacing=subband_half_width/float(n_channels_per_subband)
foffs=np.array([ch*tone_spacing for ch in range(n_channels_per_subband)])
foffs-=np.max(foffs)/2.
 
asa0=S.get_amplitude_scale_array(band)
cfa0=S.get_center_frequency_array(band)
asa=np.empty_like(asa0)
cfa=np.empty_like(cfa0)
cfa[:]=cfa0
asa[:]=asa0


subbands=np.arange(S.get_number_sub_bands())
processed_channels = S.get_processed_channels()

for sb in subbands:
    chans_in_subband=S.get_channels_in_subband(band,sb)

    # only operate on processed subbands
    if chans_in_subband[0] in processed_channels:

        cfa[chans_in_subband]=foffs
        frand=np.random.normal(0,freq_std_mhz,n_channels_per_subband)
        cfa[chans_in_subband]+=frand

        #print(cfa[chans_in_subband])
        asa[chans_in_subband]=amplitude

        if one_subband_at_a_time:
            print('-> Setting band {}, subband {}.'.format(band,sb))
            S.set_center_frequency_array(band,cfa)
            S.set_amplitude_scale_array(band,asa)        
            time.sleep(wait_btw_subbands_sec)
            #input('Press return to continue to next subband...')
        
if not one_subband_at_a_time:
    if restrict_nper_band is not None:
        print('-> Restricting nchan to %d.'%restrict_nper_band)
        import random
        assigned_channels=np.where(asa!=0)[0]
        ntotal=len(assigned_channels)
        n2kill=(ntotal-restrict_nper_band)
        channels2kill=random.sample(list(assigned_channels),n2kill)
        cfa[channels2kill]=0
        asa[channels2kill]=0
    
    S.set_center_frequency_array(band,cfa)
    S.set_amplitude_scale_array(band,asa)
