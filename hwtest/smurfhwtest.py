# This file will be comprised of functions used
# to perform Dan's tests on smurf hardware

# Have not had a chance to test any of these functions
# I still need to run Dan's server to check some things

# !!! Remember to uncomment imports when merging local and regular hwtest !!!

# from epics import caget, caput
# import time


class SetupHardware:

	# Takes in which hardware to set (attenuator or waveform)
	# Sets the location of the variable chosen
	# Saves value to set for the variable chosen

	# From Dan's code
	# This is a list of values that an attenuator can be set to
	atten_values = [0, 1, 2, 4, 8, 16, 31]

	def __init__(self, hw_to_set, hw_inst=-1):
		if hw_to_set == "ucatten":

			# Location of uc attenuators missing the server name
			self.location = ":AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:UC"

			# Checks if entered instance is within correct range
			if hw_inst in range(1,5):
				self.inst = hw_inst
			elif hw_inst == -1:
				# This is a place holder
				# If user doesn't input an instance we will assume they want to
				# update all instances of the hardware at the same time
				self.inst = hw_inst
			else:
				print("ERROR: Unrecognized instance")

		elif hw_to_set == "dcatten":

			#Location of dc attneuators missing the server name
			self.location = ":AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:DC"

			# Checks if entered instance is within correct range
			if hw_inst in range(1,5):
				self.inst = hw_inst
			elif hw_inst == -1:
				self.inst = 1
			else:
				print("ERROR: Unrecognized instance")

		elif hw_to_set == "waveform":

			# Location of waveform missing the server name and specific base number exp. Base[0]:waveformselect
			self.location = ":AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:"

			# Checks if entered instance is within correct range
			if hw_inst in range(4):
				self.inst = hw_inst
			elif hw_inst == -1:
				self.inst = 1
			else:
				print("ERROR: Unrecognized instance")

		else:
			print("Hardware entered is not currently supported")
			self.location = "LOCATION UNKNOWN"
			self.inst = hw_inst

	def set_atten(self, atten_value):

		# This allows for user to input the attenuator value at init or here.
		# We are checking to see if user enters a number

		# Here we make sure user entered attenuator value is a valid value to set
		if atten_value in SetupHardware.atten_values:
			value_to_set = atten_value
		else:
			print("Attenuator value has been set to default of 0")
			value_to_set = 0

		atten_location = self.location + "[" + str(self.inst) + "]"

		# ~~ FOR SERVER INTERFACE ~~
		# caput(atten_location, value_to_set)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", atten_location)
		print("Value to set:", value_to_set)

	def set_waveform(self, wave_value):
		wave_location = self.location + "Base[" + str(self.inst) + "]:waveformselect"

		if wave_value == 0 or wave_value == 1:
			value_to_set = wave_value
		else:
			print("Waveform has been set to default value of 0")
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
uc_attens.set_all_attens(1)

# Testing DC attenuators
print("\n")
print("Testing all DC attenuators...")
dc_attens = SetupHardware(hw_types[1])
dc_attens.set_all_attens(0)

# Testing invalid type attenuator
print("\n")
print("Testing incorrect hardware type...")
invalid_hardware = SetupHardware("iohfwehoh", hw_inst=3)
invalid_hardware.set_atten(16)

# Testing atten_values list
print("\n")
print("Testing atten_values at instance 2...")
my_atten = SetupHardware(hw_types[0], hw_inst=2)
for num in SetupHardware.atten_values:
	my_atten.set_atten(num)

# Testing all waveforms
print("\n")
print("Testing set_all_waveforms...")
all_waveforms = SetupHardware(hw_types[2])
all_waveforms.set_all_waveforms(1)

# Testing set_waveform
print("\n")
print("Testing set_waveform with instance 3 and value 0...")
my_waveform = SetupHardware(hw_types[2], hw_inst=3)
my_waveform.set_waveform(0)