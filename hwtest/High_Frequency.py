import SetHardware
from ReadData import FullBandResp
from AMC_test import set_env
import matplotlib.pyplot as plt
import time
import numpy as np


def UC_high_band_response():

	# Initializing attenuation values
	attenuation_values = SetHardware.Attenuator.acceptable_values

	for band in range(5,9):

		# Setting attenuator
		my_attenuator = SetHardware.UCAttenuator(atten_inst=band)

		# Setting up the figure
		plt.figure(band - 4)
		plt.title("High Frequency UC band " + str(band - 5) + " vs Attenuator")
		plt.xlabel("Frequency (MHz)")
		plt.ylabel("Magnitude (dB)")
		for value in attenuation_values:

			# Checking which attenuator value I am currently getting data for
			print("Testing HB UC attenuator", str(band), "value:", value)

			# Setting the attenuator value
			my_attenuator.set_value(value_to_set=value)
			time.sleep(0.1)

			# Retrieving frequency and response data from FullBandResponse
			# Band only accepts values from 0-3 so we subtract 1 from band variable
			fullResp = FullBandResp(band=band - 1)
			freqs = fullResp.freq
			resp = fullResp.resp

			# Plotting the data we just received
			freqMHz = np.divide(freqs, 1e6)
			dBresp = 20*np.log10(np.absolute(resp))
			plt.plot(freqMHz, dBresp)

		my_attenuator.set_value(value_to_set=0)
		plt.show()


if __name__ == "__main__":

	# Setting up EPICs environment
	server = 5066
	address = "134.79.219.255 172.26.97.63"
	set_env(server_port=server, address_list=address)

	# Setting all waveforms to zero
	all_waves = SetHardware.Waveform()
	all_waves.set_all_waveforms(wave_value=0)

	# Setting all attenuators to zero dB
	all_attens = SetHardware.Attenuator()
	all_attens.set_all(value_to_set=0)

	# Showing plots for high band uc attenuation
	UC_high_band_response()

else:
	print("Executed from import of High_Frequency")
