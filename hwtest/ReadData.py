from GetData import ReadStreamData
from SetHardware import DaqMux



class ReadUnknown:

	def __init__(self, inst):

		if inst > 3:
			self.bay = 1
			self.inst = inst - 4
		else:
			self.bay = 0
			self.inst = inst

		self.my_daq = DaqMux(bay=self.bay)


class ReadAdcData(ReadUnknown):

	def __init__(self, inst, datalength):
		super().__init__(inst)
		self.my_daq.set_adc_daq(adcnumber=self.inst, datalength=datalength)
		data = ReadStreamData(bay=self.bay)
		self.q_data = data.q_data
		self.i_data = data.i_data

		# I Data is the real component of Data
		# Q Data is the imaginary component of Data
		self.adc_data = []
		for index in range(len(self.q_data)):
			self.adc_data.append(complex(self.i_data[index], self.q_data[index]))


class ReadDacData(ReadUnknown):

	def __init__(self, inst, datalength):
		super().__init__(inst)
		self.my_daq.set_dac_daq(dacnumber=self.inst, datalength=datalength)
		data = ReadStreamData(bay=self.bay)
		self.q_data = data.q_data
		self.i_data = data.i_data

		# I Data is the real component of Data
		# Q Data is the imaginary component of Data
		self.dac_data = []
		for index in range(len(self.q_data)):
			self.dac_data.append(complex(self.i_data[index], self.q_data[index]))