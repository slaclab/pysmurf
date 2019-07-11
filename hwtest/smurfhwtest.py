# This file will be comprised of functions used
# to perform Dan's tests on smurf hardware

# Have not had a chance to test any of these functions
# I still need to run Dan's server to check some things

# !!! Remember to uncomment epics import when merging local and regular hwtest !!!

# from epics import caget, caput
import time


class SetupHardware:

	# Takes in which hardware to set (attenuator or waveform)
	# Sets the location of the variable chosen
	# Saves value to set for the variable chosen

	def __init__(self, hw_value, hw_to_set):
		if hw_to_set == "atten":

			# Location of attenuators missing the server name and specific attenuator exp. UC[1]
			self.location = ":AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:"
		elif hw_to_set == "waveform":

			# Location of waveform missing the server name and specific base number exp. Base[0]:waveformselect
			self.location = ":AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:"
		else:
			print("Hardware entered is not currently supported")

		self.hw_value = hw_value

	# def set_all_waveforms(self, wave_value):
	#
	# 	# Don't know where to find wave_form_server
	# 	# I need to run the gui dan showed me to get a better idea
	# 	# Should end the string with a colon before the word Base
	# 	wave_form_server = self.location_all
	#
	# 	# Need to set all four Base waveforms
	# 	for index in range(4):
	# 		this_waveform = wave_form_server + "Base[" + str(index) + "]:waveformselect"
	# 		caput(this_waveform, wave_value)

	# def set_all_uc_atten(self, atten_value):
	#
	# 	# I don't know where to find uc_atten_server
	# 	# I need to run gui dan showed me to get a better idea
	# 	# Should end the string with 'ATT:'
	#
	# 	atten_server = self.location_all
	# 	for index in range(4):
	# 		this_atten = atten_server + "UC[" + str(index) + "]"
	# 		caput(this_atten, atten_value)
	#
	# 		# Need to pause after setting each attenuator
	# 		time.sleep(0.1)

	# def set_all_dc_atten(self, atten_value):
	#
	# 	# I don't know where to find dc_atten_server
	# 	# I need to run gui dan showed me to get a better idea
	# 	# Should end the string with 'ATT:'
	# 	atten_server = self.location_all
	#
	# 	for index in range(4):
	# 		this_atten = atten_server + "DC[" + str(index) + "]"
	# 		caput(this_atten, atten_value)
	#
	# 		# Need to pause after setting each attenuator
	# 		time.sleep(0.1)

all_attens = SetupHardware(0, "atten")
print("This is the address for all attenuators: ", all_attens.location_all)