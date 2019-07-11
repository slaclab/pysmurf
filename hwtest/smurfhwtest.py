# This file will be comprised of functions used
# to perform Dan's tests on smurf hardware

# Have not had a chance to test any of these functions
# I still need to run Dan's server to check some things

from epics import caget, caput
import time

class setup():

	def set_all_waveforms(self, wave_value):

		# Don't know where to find wave_form_server
		# I need to run the gui dan showed me to get a better idea
		# Should end the string with a colon before the word Base
		wave_form_server = "Location of waveform variables in Dan's server"

		# Need to set all four Base waveforms
		for index in range(4):
			this_waveform = wave_form_server + "Base[" + str(index) + "]:waveformselect"
			caput(this_waveform, wave_value)

	def set_all_uc_atten(self, atten_value):

		# I don't know where to find uc_atten_server
		# I need to run gui dan showed me to get a better idea
		# Should end the string with 'ATT:'
		atten_server = "Location of attenuators"
		for index range(4):
			this_atten = atten_server + "UC[" + str(index) + "]"
			caput(this_atten, atten_value)

			# Need to pause after setting each attenuator
			time.sleep(0.1)

	def set_all_dc_atten(self, atten_value):

		# I don't know where to find dc_atten_server
		# I need to run gui dan showed me to get a better idea
		# Should end the string with 'ATT:'
		atten_server = "Location of attenuators"

		for index range(4):
			this_atten = atten_server + "DC[" + str(index) + "]"
			caput(this_atten, atten_value)

			# Need to pause after setting each attenuator
			time.sleep(0.1)