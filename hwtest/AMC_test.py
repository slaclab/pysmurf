from ReadData import FullBandResp
import SetHardware
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import numpy
import datetime
import os


"""
This file is designed to follow AMC_test.m written by Dan Van Winkle.
The goal here is to receive the same output as AMC_test, but using Python all written by myself.

"""


def set_env(server_port, address_list):
	"""
	This function should set up the EPICs environment so that the user doesn't need to input export commands

	Args:
	server_port: This variable should tell this function which server port to set EPICs to
	address_list: This variable should tell EPICs which address we are using to talk to the server

	:return: This function does not return anything, but rather sets up the EPICs environment
	"""
	os.environ["EPICS_CA_SERVER_PORT"] = str(server_port)
	os.environ["EPICS_CA_ADDR_LIST"] = address_list


def get_time():
	# Creating 'current_timestamp' for use of current time
	datetime_object = str(datetime.datetime.now())
	today, current_time = datetime_object.split(' ')
	time_seconds, time_milliseconds = current_time.split('.')
	hour, minute, second = time_seconds.split(':')
	better_time_seconds = hour + '.' + minute + '.' + second
	current_timestamp = today + '_' + better_time_seconds
	return current_timestamp


def save_pdf(time_string, figures, name):
	# Saving figures on separate pdf pages
	# path is the path to the images folder where I'm storing the pdf
	path = "/afs/slac.stanford.edu/u/gu/shadduck/cryo_det/images/"
	filename = name + time_string + '.pdf'
	with PdfPages(path + filename) as pp:
		for fig in figures:
			pp.savefig(fig)

	print("File Location:", path + filename)


def converter_vs_attenuator(atten_type):
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

	:param atten_type: Either UC or DC to specify which type of attenuator we are testing
	:return: atten_fig: a figure that contains a subplot of 4 attenuator tests
	"""
	# Initializing attenuation values
	attenuation_values = SetHardware.Attenuator.acceptable_values

	atten_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	sublot_list = [ax1, ax2, ax3, ax4]
	atten_fig.suptitle(atten_type + "-Converter Response vs Atten")

	for band in range(1, 5):

		sub_fig = sublot_list[band-1]
		# Setting up our figure. Should be 4 figures with 8 plots on each
		# We use band-1 for the title because that's how they are labeled above
		# We use band for figure number because we always start with figure 1
		sub_fig.set_title("Band " + str(band - 1))
		sub_fig.axis([-250, 250, -40, 20])

		# Sets the attenuator type at the desired band
		if atten_type == "UC":
			atten = SetHardware.UCAttenuator(atten_inst=band)
		elif atten_type == "DC":
			atten = SetHardware.DCAttenuator(atten_inst=band)
		else:
			atten = SetHardware.UCAttenuator(atten_inst=band)

		for atten_value in attenuation_values:
			# Checking which attenuator value I am currently getting data for
			print("Testing", atten_type + " attenuator", str(band), "value:", atten_value)

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
			dBresp = 20*numpy.log10(numpy.absolute(resp))
			sub_fig.plot(freqMHz, dBresp)

		# Resets current attenuator value to zero before setting next attenuator
		atten.set_value(value_to_set=0)

	# Setting Axis Labels for the subplot
	for ax in sublot_list:
		ax.set(xlabel="frequency (MHz)", ylabel="Response (dB)")

	# Setting labels to outside so each subplot doesn't have extra labels
	for ax in atten_fig.get_axes():
		ax.label_outer()

	return atten_fig


if __name__ == "__main__":

	# Setting EPICs environment
	server = 5066
	address = "134.79.219.255 172.26.97.63"
	set_env(server_port=server, address_list=address)

	# Setting all waveforms to zero
	all_waves = SetHardware.Waveform()
	all_waves.set_all_waveforms(wave_value=0)

	# Setting all attenuators to zero dB
	all_attens = SetHardware.Attenuator()
	all_attens.set_all(value_to_set=0)

	# Initializing a total figures list
	total_figs = []

	# Appending figures from up converter test to total figures list
	up_converter_subplot = converter_vs_attenuator(atten_type="UC")
	total_figs.append(up_converter_subplot)

	# Appending figures from down converter test to total figures list
	down_converter_subplot = converter_vs_attenuator(atten_type="DC")
	total_figs.append(down_converter_subplot)

	# Getting timestamp for pdf filename
	current_time = get_time()

	# Saving figures to pdf
	filename = "Both_Converters_Response"
	save_pdf(name=filename, time_string=current_time, figures=total_figs)

else:
	print("Executed from import of AMC_test")
