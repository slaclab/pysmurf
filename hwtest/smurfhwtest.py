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

		atten_check = atten_type.upper()

		# Checks if proper attenuator type is entered
		if atten_check == "UC" or atten_check == "DC":
			self.atten_location = self.location + atten_check + "[" + str(atten_num) + "]"

			# ~~ FOR SERVER INTERFACE ~~
			# caput(self.location, self.hw_value)
			# time.sleep(0.1)

			# ~~ FOR LOCAL TESTING ~~
			print("Variable location:", self.atten_location)
			print("Value to set:", self.hw_value)

		else:
			print("ERROR: Attenuator type not recognized")

	def set_waveform(self, wave_num):
		self.wave_location = self.location + "Base[" + str(wave_num) + "]:waveformselect"

		# ~~ FOR SERVER TESTING ~~
		# caput(self.location, self.hw_value)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable Location:", self.wave_location)
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

# Variables to use for local testing
# __________________________________

# Testing UC attenuators
print("\n")
print("Testing all UC attenuators...")
uc_attens = SetupHardware(0, "atten")
uc_attens.set_all_attens("uc")

# Testing DC attenuators
print("\n")
print("Testing all DC attenuators...")
dc_attens = SetupHardware(1, "atten")
dc_attens.set_all_attens("dc")

# Testing invalid type attenuator
print("\n")
print("Testing incorrect attenuator type...")
invalid_attens = SetupHardware(3, "atten")
invalid_attens.set_atten("lc", 2)
