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


def get_time():
	# Creating 'current_timestamp' for use of current time
	datetime_object = str(datetime.datetime.now())
	today, current_time = datetime_object.split(' ')
	time_seconds, time_milliseconds = current_time.split('.')
	hour, minute, second = time_seconds.split(':')
	better_time_seconds = hour + '.' + minute + '.' + second
	current_timestamp = today + '_' + better_time_seconds
	return current_timestamp


def save_pdf(time_string, figures):
	# Saving figures on separate pdf pages
	# path is the path to the images folder where I'm storing the pdf
	path = "/afs/slac.stanford.edu/u/gu/shadduck/cryo_det/images/"
	filename = 'Up_Converter_Response_' + time_string + '.pdf'
	with PdfPages(path + filename) as pp:
		for fig in figures:
			pp.savefig(fig)

	print("File Location:", path + filename)


def converter_vs_attenuator(atten_type, attenuation_values):
	"""
	This function should return 4 figures with 7 lines each that represent different attenuation values
	Each new figure represents another attenuator band being tested

	The variable atten_type is used to denote if we wish to run Up converter or down converter
	Default for atten_type is 'UC'

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

	:param attenuation_values: a list of acceptable values that the attenuators can be set to
	:return: atten_figs: a list that contains all the figures from performing up converter testing
	"""
	atten_figs = []
	for band in range(1, 5):

		# Setting up our figure. Should be 4 figures with 8 plots on each
		# We use band-1 for the title because that's how they are labeled above
		# We use band for figure number because we always start with figure 1
		atten_figs.append(plt.figure(band))
		plt.title("Band " + str(band - 1) + " Up-Converter Response vs Atten")
		plt.xlabel("frequency (MHz)")
		plt.ylabel("Response (dB)")
		plt.axis([-250, 250, -90, 10])
		legend_list = []

		# Sets the attenuator type at the desired band
		if atten_type == "UC":
			atten = SetHardware.UCAttenuator(atten_inst=band)
		elif atten_type == "DC":
			atten = SetHardware.DCAttenuator(atten_inst=band)
		else:
			atten = SetHardware.UCAttenuator(atten_inst=band)

		# Setting noiseSelect to correct band
		noisepv = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base[" + str(band-1) + "]:noiseSelect"
		caput(noisepv, 1)

		# There is a lot of pause time to allow for attenuators to adjust
		# Adding print statements so I can see where I am in the program
		print("Testing attenuator band:", band - 1)

		for atten_value in attenuation_values:
			# Checking which attenuator value I am currently getting data for
			print("Testing attenuator value:", atten_value)

			# Setting desired attenuator value
			atten.set_value(value_to_set=atten_value)
			# Attenuators need delay after setting
			time.sleep(0.1)

			# Retrieving frequency and response data from FullBandResponse
			# Band only accepts values from 0-3 so we subtract 1 from band variable
			fullResp = FullBandResp(band=band - 1)
			freqs = fullResp.freq
			resp = fullResp.resp

			# Plotting the data we just collected
			freqMHz = [x / (1e6) for x in freqs]
			dBresp = [20 * math.log10(abs(x)) for x in resp]
			plt.plot(freqMHz, dBresp)

			# Setting a legend for each line in list titled legend
			legend_list.append("Attenuator value = " + str(atten_value))

		# Removing noiseSelect from current band
		caput(noisepv, 0)

		plt.legend(legend_list, loc="lower center")
		atten.set_value(value_to_set=0)

	return atten_figs


if __name__ == "__main__":

	# Setting all waveforms to zero
	all_waves = SetHardware.Waveform()
	all_waves.set_all_waveforms(wave_value=0)

	# Setting all attenuators to zero dB
	all_attens = SetHardware.Attenuator()
	all_attens.set_all(value_to_set=0)

	# All values to set attenuators to
	atten_values = SetHardware.Attenuator.acceptable_values

	# Appending figures from up converter test to total figures list
	up_converter_figs = converter_vs_attenuator(atten_type="UC", attenuation_values=atten_values)

	# Getting timestamp for pdf filename
	current_time = get_time()

	# Saving figures to pdf
	save_pdf(time_string=current_time, figures=up_converter_figs)

else:
	print("Executed from import of AMC_test")
