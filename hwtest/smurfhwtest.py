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
			self.location = "LOCATION UNKNOWN"

		self.hw_value = hw_value

	def set_atten(self, atten_type, atten_num):
		self.location = self.location + atten_type.upper() + "[" + str(atten_num) + "]"

		# ~~ FOR SERVER INTERFACE ~~
		# caput(self.location, self.hw_value)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", self.location)
		print("Value to set:", self.hw_value)

	def set_waveform(self, wave_num):
		self.location = self.location + "Base[" + str(wave_num) + "]:waveformselect"

		# ~~ FOR SERVER TESTING ~~
		# caput(self.location, self.hw_value)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable Location:", self.location)
		print("Value to set:", self.hw_value)

	def set_all_waveforms(self):

		# Need to set all four Base waveforms
		for index in range(4):
			self.set_waveform(index)

	def set_all_attens(self, atten_type):

		# index 1-4 because attenuators are labeled 1-4 not 0-3 like waveforms
		# Use for loop to iterate through all attenuators
		for index in range(1, 5):
			self.set_atten(atten_type, index)


all_attens = SetupHardware(0, "atten")
all_attens.set_all_attens("uc")