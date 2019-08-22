from SetHardware import SetHwTrigger
from epics import caget, caput
import numpy as np
import time


class ReadStreamData:
	"""
	This class sets the streams to be read based on the input.
	This will return two lists of data in i and q data form

	Args:
		bay: this variable tells which streams to read for data collection. Accepted values are 0 and 1 only

	Returns:
		self.i_data: the real component of the data read from the stream
		self.q_data: the imaginary component of the data read from the stream
	"""
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

	def get_new_data(self):
		"""
		This function uses the hardware trigger to make sure that new i and q data is available to be read.
		It does this by setting the hardware trigger and checking the new data received against the old data.
		If new i and q data is received, the function stops and returns the new data.

		:param
			None
		:return:
			new_qdata: this is the new q data read by the desired stream
			new_idata: this is the new i data read by the desired stream
		"""
		old_qdata = caget(self.q_stream)
		old_idata = caget(self.i_stream)
		new_qdata = caget(self.q_stream)
		new_idata = caget(self.i_stream)

		for number in range(20):
			new_qdata = caget(self.q_stream)
			new_idata = caget(self.i_stream)

			if np.array_equal(old_qdata, new_qdata) or np.array_equal(old_idata, new_idata):
				# print("No new data")
				# print("Running HwTrigger...")
				SetHwTrigger()
			else:
				# print("New data received!")
				break

			time.sleep(0.1)

		return new_qdata, new_idata


if __name__ == "__main__":
	# Testing StreamData class and idata monitor function
	print("Testing ReadStreamData...")
	data = ReadStreamData(bay=0)
	idata = data.i_data
	qdata = data.q_data
	print("I Data:", idata, "\nQ Data:", qdata)

else:
	print("Executed from import of GetData")
