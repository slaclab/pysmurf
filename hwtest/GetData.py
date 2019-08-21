from SetHardware import SetHwTrigger
from epics import caget, caput
import numpy as np
import time


class ReadStreamData:

	def __init__(self, bay):

		if bay == 0:

			# Written as stream0 in matlab readStreamData.m
			# Represents the imaginary component of complex phasor data
			self.q_stream = 'dans_epics:AMCc:Stream0'

			# Written as stream1 in matlab code readStreamData.m
			# This data represents the real component in complex phasor data
			self.i_stream = 'dans_epics:AMCc:Stream1'

		elif bay == 1:

			# Written as stream0 in matlab readStreamData.m
			# Represents the imaginary component of complex phasor data
			self.q_stream = 'dans_epics:AMCc:Stream4'

			# Written as stream1 in matlab code readStreamData.m
			# This data represents the real component in complex phasor data
			self.i_stream = 'dans_epics:AMCc:Stream5'

		else:
			print("ERROR: Bay unrecognized. Set to default 0")
			self.q_stream = 'dans_epics:AMCc:Stream0'
			self.i_stream = 'dans_epics:AMCc:Stream1'

		# ~~ Returning new q_data and new i_data ~~
		self.q_data, self.i_data = self.get_new_data()

	def get_new_data(self, show=False):

		old_qdata = caget(self.q_stream)
		old_idata = caget(self.i_stream)
		new_qdata = caget(self.q_stream)
		new_idata = caget(self.i_stream)

		for number in range(20):
			new_qdata = caget(self.q_stream)
			new_idata = caget(self.i_stream)

			if np.array_equal(old_qdata, new_qdata) or np.array_equal(old_idata, new_idata):
				print("No new data")
				print("Running HwTrigger...")
				SetHwTrigger()
			else:
				print("New data received!")
				break

			time.sleep(0.25)

		if show is True:
			print("Q_Data:", new_qdata, "\n" + "I_Data:", new_idata)

		return new_qdata, new_idata


if __name__ == "__main__":
	# Testing StreamData class and idata monitor function
	data = ReadStreamData(bay=0)
	idata = data.i_data
	qdata = data.q_data
	print("I Data:", i_data, "\nQ Data:", q_data)

else:
	print("Executed from import of GetData")
