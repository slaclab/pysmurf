from SetHardware import SetHwTrigger
from epics import caget, caput, camonitor, camonitor_clear
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

	"""
	def monitor_idata(self):
		# Extract the value passed to i_data from the monitor

		# ~~For use in Server~~
		new_idata_list = []
		camonitor(self.i_stream, writer=lambda arg: new_idata_list.append(arg))

		# grabs the data from the first monitor string
		# in my testing, camonitor will write to list upon initialization
		# sometimes it will write the same value twice upon initialization
		previous_data = new_idata_list[0].split(' ')[-1]

		# Checks if data received is different than the initialized data
		# While loop will continue checking for new data until self.get_new_idata is no longer None
		while self.get_new_idata is None:

			# ~~For use in Server~~
			# Not sure if this line below is necessary
			# camonitor(self.i_stream, writer=lambda arg: new_idata_list.append(arg))

			for string in new_idata_list:
				data = string.split(' ')[-1]
				if data == previous_data:
					# This would occur if camonitor initializes two or more strings with same initial value
					self.get_new_idata = None
					print("No new idata")
					print("Current data:", data)
					print("Previous Data:", previous_data)
				else:
					# once we get new data this statement should execute
					self.get_new_idata = 1
					print("New idata received")
					print("New I data:", data)
			time.sleep(0.1)

		camonitor_clear(self.i_stream)

	def monitor_qdata(self):
		# Extract the value passed to q_data from the monitor

		# ~~For use in Server~~
		new_qdata_list = []
		camonitor(self.q_stream, writer=lambda arg: new_qdata_list.append(arg))

		previous_data = new_qdata_list[0].split(' ')[-1]

		while self.get_new_qdata is None:

			# ~~For use in Server~~
			# Not sure if this line below is necessary
			# camonitor(self.q_stream, writer=lambda arg: new_qdata_list.append(arg))

			for string in new_qdata_list:
				data = string.split(' ')[-1]
				if data == previous_data:
					self.get_new_qdata = None
					print("No new qdata")
					print("Current data:", data)
					print("Previous data:", previous_data)
				else:
					self.get_new_qdata = 1
					print("New qdata received")
					print("New Q data:", data)
			time.sleep(0.1)

		camonitor_clear(self.q_stream)

	def wait_data(self):

		while self.get_new_idata is None or self.get_new_qdata is None:
			time.sleep(0.1)

		# ~~For use in Server~~
		self.idata = caget(self.i_stream)
		self.qdata = caget(self.q_stream)

		return self.idata, self.qdata
	"""

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
	data = StreamData(bay=0)
	idata = data.i_data
	qdata = data.q_data
	print("I Data:", i_data, "\nQ Data:", q_data)

else:
	print("GetData accessed from import")
