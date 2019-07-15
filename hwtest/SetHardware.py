# from epics import caget, caput
# import time

# I think the server name is 'dans_epics'


class Attenuator:

	# This is a list of values that an attenuator can be set to
	acceptable_values = [0, 1, 2, 4, 8, 16, 31]

	def __init__(self, atten_inst=-1):

		# Sets general location for all attenuators
		# Will need to specify UC or DC in specific attenuator Classes
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:"
		self.inst = atten_inst

	def set_value(self, value_to_set):
		# This function will set the atten_value to the self.location

		# Here we make sure user entered attenuator value is a valid value to set
		if value_to_set not in Attenuator.acceptable_values:
			print("ERROR: Attenuator value invalid. Value has been set to default of 0")
			value_to_set = 0

		if self.inst == -1:
			print("ERROR: Attenuator instance not specified. Instance set to default 1")
			self.inst = 1

		if self.inst not in range(1, 5):
			print("ERROR: Attenuator instance is out of range. Instance set to default 1")
			self.inst = 1

		atten_location = self.location + "[" + str(self.inst) + "]"

		# ~~ FOR SERVER INTERFACE ~~
		# caput(atten_location, value_to_set)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", atten_location)
		print("Value to set:", value_to_set)

	def set_all(self, value_to_set):
		# Sets all instances of one type of attenuator to same value

		for num in range(1, 5):
			self.inst = num
			self.set_value(value_to_set)


class UCAttenuator(Attenuator):
	# This class will inherit from the Attenuator base class

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "UC"


class DCAttenuator(Attenuator):
	# This class will inherit from the Attenuator base class

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "DC"


class Waveform:

	def __init__(self, waveform_inst=-1):
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base"
		self.inst = waveform_inst

	def set_value(self, wave_value):
		# This function will set wave_value to the instance specified in init function
		# or by the set all function for all waveforms

		# Here we are checking that all requirements are satisfied to set a value
		if wave_value not in [0, 1]:
			print("ERROR: Wave value invalid. Value set to default 0")
			wave_value = 0

		if self.inst == -1:
			print("ERROR: Waveform instance is not defined. Set to default 0")
			self.inst = 0

		if self.inst not in range(4):
			print("ERROR: Waveform instance is out of range. Instance set to default 0")
			self.inst = 0

		wave_location = self.location + "[" + str(self.inst) + "]:waveformSelect"

		# ~~ FOR SERVER INTERFACE ~~
		# caput(wave_location, wave_value)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", wave_location)
		print("Value to set:", wave_value)

	def set_all_waveforms(self, wave_value):

		for num in range(4):
			self.inst = num
			self.set_value(wave_value)

# Variables to use for local testing
# __________________________________

# Testing UC attenuator
print("\n")
print("Testing set_attenuator function...")
my_attenuator = UCAttenuator(atten_inst=4)
my_attenuator.set_value(value_to_set=8)

# Testing set_all
print("\n")
print("Testing set_all function...")
all_attenuators = UCAttenuator()
all_attenuators.set_all(value_to_set=2)

# Testing DC attenuator
print("\n")
print("Testing DC attenuator class...")
dc_attenuator = DCAttenuator(atten_inst=1)
dc_attenuator.set_value(value_to_set=16)

# Testing waveform class
print("\n")
print("Testing Waveform class...")
my_waveform = Waveform(waveform_inst=0)
my_waveform.set_value(wave_value=1)

# Testing invalid waveform value
print("\n")
print("Testing invalid waveform value entry")
invalid_waveform = Waveform(waveform_inst=1)
invalid_waveform.set_value(wave_value=9)

# Testing invalid waveform instance
print("\n")
print("Testing invalid waveform instance...")
invalid_instance_wave = Waveform(waveform_inst=5)
invalid_instance_wave.set_value(wave_value=3)

# Testing invalid attenuator value and instance
print("\n")
print("Testing invalid instance and value for attenuator class...")
wack_attenuator = UCAttenuator(atten_inst=6)
wack_attenuator.set_value(value_to_set=40)

# Testing set_all_waveforms
print("\n")
print("Testing set_all_waveforms function...")
all_waveforms = Waveform()
all_waveforms.set_all_waveforms(wave_value=1)