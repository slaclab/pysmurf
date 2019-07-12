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

		elif self.inst == -1:
			print("ERROR: Attenuator instance not specified. Instance set to default 1")
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


# Variables to use for local testing
# __________________________________

# Testing attenuator
print("\n")
print("Testing set_attenuator function...")
my_attenuator = UCAttenuator()
my_attenuator.set_value(value_to_set=8)

# Testing set_all
print("\n")
print("Testing set_all function...")
all_attenuators = UCAttenuator()
all_attenuators.set_all(value_to_set=2)
