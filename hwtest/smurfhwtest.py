# This file will be comprised of functions used
# to perform Dan's tests on smurf hardware

# I still need to run Dan's server to check some things
# Dan said server name might be 'dans_epics'
# Will use this in server_name for now until I find it doesn't work

# !!! Remember to uncomment imports when merging local and regular hwtest !!!

# from epics import caget, caput
# import time


class SetupHardware:

	# Takes in which hardware to set (attenuator or waveform)
	# Sets the location of the variable chosen

	# From Dan's code

	# This is a list of values that an attenuator can be set to
	atten_values = [0, 1, 2, 4, 8, 16, 31]

	# This is the name of the server as specified by Dan
	server_name = "dans_epics"

	def __init__(self, hw_to_set, hw_inst=-1):
		if hw_to_set == "ucatten":

			# Location of uc attenuators
			self.location = SetupHardware.server_name + ":AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:UC"

		elif hw_to_set == "dcatten":

			# Location of dc attneuators
			self.location = SetupHardware.server_name + ":AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:DC"

		elif hw_to_set == "waveform":

			# Location of waveform missing specific base number exp. Base[0]:waveformselect
			self.location = SetupHardware.server_name + ":AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:"

		else:
			self.location = "LOCATION UNKNOWN"

		self.inst = hw_inst

	def set_atten(self, atten_value):

		# This allows for user to input the attenuator value at init or here.
		# We are checking to see if user enters a number

		# Here we make sure user entered attenuator value is a valid value to set
		if atten_value in SetupHardware.atten_values:
			value_to_set = atten_value
		else:
			print("ERROR: Attenuator value invalid. Value has been set to default of 0")
			value_to_set = 0

		# Checking if all conditions are satisfied to write value to desired location
		if self.inst == -1:
			# -1 is the default for init function
			# We return an error because the user did not specify which instance
			# to write this value to
			print("ERROR: Attenuator instance was not specified")

		elif self.inst not in range(1, 5):
			# Covers any instance that is not in proper range
			print("ERROR: Attenuator instance invalid")

		elif "dans_epics" not in self.location:
			# This is true if hardware entered is not supported
			print("ERROR: Unrecognized hardware name")

		else:

			# If instance is properly defined, we are able to
			# write a value to the attenuator variable
			atten_location = self.location + "[" + str(self.inst) + "]"

			# ~~ FOR SERVER INTERFACE ~~
			# caput(atten_location, value_to_set)
			# time.sleep(0.1)

			# ~~ FOR LOCAL TESTING ~~
			print("Variable location:", atten_location)
			print("Value to set:", value_to_set)

	def set_waveform(self, wave_value):

		# Checking if all conditions are satisfied to write value to desired location
		if self.inst == -1:
			# -1 is the default for init function
			# We return an error because the user did not specify which instance
			# to write this value to
			print("ERROR: Waveform instance was not specified")

		elif self.inst not in range(4):
			# Covers any inst that is not in proper range
			print("ERROR: Waveform instance entered is invalid")

		elif "dans_epics" not in self.location:
			# This is true if hardware entered is not supported
			print("ERROR: Unrecognized hardware name")

		else:

			# If instance is properly specified, we are able to
			# write the value to waveform variable
			wave_location = self.location + "Base[" + str(self.inst) + "]:waveformSelect"

			if wave_value == 0 or wave_value == 1:
				value_to_set = wave_value
			else:
				print("ERROR: Waveform value invalid. Value has been set to default of 0")
				value_to_set = 0

			# ~~ FOR SERVER TESTING ~~
			# caput(wave_location, value_to_set)

			# ~~ FOR LOCAL TESTING ~~
			print("Variable Location:", wave_location)
			print("Value to set:", value_to_set)

	def set_all_waveforms(self, wave_value):

		# Need to set all four Base waveforms
		for index in range(4):
			self.inst = index
			self.set_waveform(wave_value)

	def set_all_attens(self, atten_value):

		# index 1-4 because attenuators are labeled 1-4 not 0-3 like waveforms
		# Use for loop to iterate through all attenuators
		for index in range(1, 5):
			self.inst = index
			self.set_atten(atten_value)


# Variables to use for local testing
# __________________________________

# Ensures that supported hardware type is selected
hw_types = ["ucatten", "dcatten", "waveform"]

# Testing UC attenuators
print("\n")
print("Testing all UC attenuators...")
uc_attens = SetupHardware(hw_types[0])
uc_attens.set_all_attens(atten_value=1)

# Testing DC attenuators
print("\n")
print("Testing all DC attenuators...")
dc_attens = SetupHardware(hw_types[1])
dc_attens.set_all_attens(atten_value=0)

# Testing invalid type attenuator
print("\n")
print("Testing incorrect hardware type...")
invalid_hardware = SetupHardware("iohfwehoh", hw_inst=3)
invalid_hardware.set_atten(atten_value=16)

# Testing atten_values list
print("\n")
print("Testing atten_values at instance 2...")
my_atten = SetupHardware(hw_types[0], hw_inst=2)
for num in SetupHardware.atten_values:
	my_atten.set_atten(atten_value=num)

# Testing all waveforms
print("\n")
print("Testing set_all_waveforms...")
all_waveforms = SetupHardware(hw_types[2])
all_waveforms.set_all_waveforms(wave_value=1)

# Testing set_waveform
print("\n")
print("Testing set_waveform with instance 3 and value 0...")
my_waveform = SetupHardware(hw_types[2], hw_inst=3)
my_waveform.set_waveform(wave_value=0)

# Testing what happens when instance is forgotten for a single attenuator
print("\n")
print("Testing unspecified uc attenuator instance...")
unknown_atten = SetupHardware(hw_types[0])
unknown_atten.set_atten(atten_value=8)

# Testing what happens when unexpected value is passed to set_atten
print("\n")
print("Testing invalid atten_value passed to set_atten")
wrong_value_atten = SetupHardware(hw_types[0], hw_inst=0)
wrong_value_atten.set_atten(atten_value=100)
