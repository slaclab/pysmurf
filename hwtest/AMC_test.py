from ReadData import FullBandResp
import SetHardware
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import math
import datetime

"""
This file is designed to follow AMC_test.m written by Dan Van Winkle.
The goal here is to receive the same output as AMC_test, but using Python all written by myself.

"""

# Creating a timestamp for current time
datetime_object = str(datetime.datetime.now())
today, current_time = datetime_object.split(' ')
time_seconds, time_milliseconds = current_time.split('.')
hour, minute, second = time_seconds.split(':')
better_time_seconds = hour + '.' + minute + '.' + second
current_timestamp = today + '_' + better_time_seconds

# Setting all waveforms to zero
all_waves = SetHardware.Waveform()
all_waves.set_all_waveforms(wave_value=0)

# Setting all attenuators to zero dB
all_attens = SetHardware.Attenuator()
all_attens.set_all(value_to_set=0)

# All values to set attenuators to
attenuation_values = SetHardware.Attenuator.acceptable_values

"""
For attenuator setting use the atten_vals

Band0 is 4-4.5 GHz uses atten_val 1
Band1 is 4.5-5 GHz uses atten_val 2
Band2 is 5-5.5 GHz uses atten_val 3
Band3 is 5.5-6 GHz uses atten_val 4

For using FullBandResp use band_vals

Band0 is 4-4.5 GHz uses band_val 0
Band1 is 4.5-5 GHz uses band_val 1
Band2 is 5-5.5 GHz uses band_val 2
Band3 is 5.5-6 GHz uses band_val 3
"""


atten_figs = []
for band in range(1, 5):

	# Setting up our figure. Should be 4 figures with 8 plots on each
	# We use band-1 for the title because that's how they are labeled above
	# We use band for figure number because we always start with figure 1
	atten_figs.append(plt.figure(band))
	plt.title("Band " + str(band-1) + " Up-Converter Response vs Atten")
	plt.xlabel("frequency (MHz)")
	plt.ylabel("Response (dB)")
	legend_list = []

	# There is a lot of pause time to allow for attenuators to adjust
	# Adding print statements so I can see where I am in the program
	print("Testing attenuator band:", band - 1)

	for atten_value in attenuation_values:

		# Checking which attenuator value I am currently getting data for
		print("Testing attenuator value:", atten_value)
		
		# Sets the attenuator at the desired band and attenuator value
		uc_atten = SetHardware.UCAttenuator(atten_inst=band)
		uc_atten.set_value(value_to_set=atten_value)
		# Attenuators need delay after setting
		time.sleep(0.1)

		# Retrieving frequency and response data from FullBandResponse
		# Band only accepts values from 0-3 so we subtract 1 from band variable
		fullResp = FullBandResp(band=band-1)
		freqs = fullResp.freq
		resp = fullResp.resp

		# Plotting the data we just collected
		freqMHz = [x/(1e6) for x in freqs]
		dBresp = [20*math.log10(abs(x)) for x in resp]
		plt.plot(freqMHz, dBresp)

		# Setting a legend for each line in list titled legend
		legend_list.append("Attenuator value = " + str(atten_value))

	plt.legend(legend_list, loc="lower right")


# Saving figures on separate pdf pages
# path is the path to the images folder where I'm storing the pdf
path = "/afs/slac.stanford.edu/u/gu/shadduck/cryo_det/images/"
filename = 'Up_Converter_Response_' + current_timestamp + '.pdf'
with PdfPages(path + filename) as pp:
	for fig in atten_figs:
		pp.savefig(fig)

print("File Location:", path + filename)
