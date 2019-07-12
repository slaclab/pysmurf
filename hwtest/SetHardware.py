# from epics import caget, caput
# import time

# I think the server name is 'dans_epics'


class Attenuator:

	# This is a list of values that an attenuator can be set to
	acceptable_values = [0, 1, 2, 4, 8, 16, 31]

	def __init__(self):

		# Sets general location for all attenuators
		# Will need to specify UC or DC in specific attenuator Classes
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:"

	def set_value(self, atten_value):
		# This function will set the atten_value to the self.location

		# Here we make sure user entered attenuator value is a valid value to set
		if atten_value in Attenuator.acceptable_values:
			value_to_set = atten_value
		else:
			print("ERROR: Attenuator value invalid. Value has been set to default of 0")
			value_to_set = 0

		# ~~ FOR SERVER INTERFACE ~~
		# caput(atten_location, value_to_set)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", self.location)
		print("Value to set:", value_to_set)


# Variables to use for local testing
# __________________________________

# Testing attenuator
print("\n")
print("Testing set_attenuator function...")
my_attenuator = Attenuator()
my_attenuator.set_value(atten_value=8)