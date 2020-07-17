print('-> Running cryocard_c03_setup.py ...')

# have to explicitly enable and set the Vd control DACs
print('-> Enabling RTM DAC32, which drives DAC_In_50k_d')
S.set_rtm_slow_dac_enable(32,2)
print('-> Setting DAC_In_50k_d to 0.65V for 50kVd=5.003')
S.set_rtm_slow_dac_volt(32,0.65) #corresponds to 50kVd->5.003

print('-> Enabling RTM DAC31, which drives DAC_In_H_d')
S.set_rtm_slow_dac_enable(31,2)
print('-> Setting DAC_In_H_d to 6V for HVd=0.53')
S.set_rtm_slow_dac_volt(31,6) #corresponds to HVd->0.53

print('Done running cryocard_c03_setup.py.')
