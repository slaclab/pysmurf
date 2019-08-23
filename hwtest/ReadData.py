from GetData import ReadStreamData
from SetHardware import DaqMux
from epics import caget, caput
import time
from scipy import signal
import math
import matplotlib.pyplot as plt


class ReadUnknown:
	"""
	This is a parent class for both ReadAdcData and ReadDacData

	Args:
		inst: should correspond to the instance or band of attenuator we are testing in main program
	Returns:
		self.bay: This attribute is to be used by ReadAdcData and ReadDacData for setting daqMux
	"""
	def __init__(self, inst):

		if inst > 3:
			self.bay = 1
			self.inst = inst - 4
		else:
			self.bay = 0
			self.inst = inst

		self.my_daq = DaqMux(bay=self.bay)


class ReadAdcData(ReadUnknown):
	"""
	Inherits some characteristics from ReadUnknown class
	Reads the complex data from 2 streams after setting ADC daqMux

	Inherits:
		self.bay: this variable describes the bay that the card we are testing is in
	Args:
		inst: this variable is described in ReadUnknown and is not acted on by this class
		datalength: describes the size of the buffer we are setting. Because we set datalength in 32 bit
					words and we read stream data in 16 bit, we will see len(q_data) is twice datalength variable
	"""
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
	"""
	Inherits some characteristics from ReadUnknown class
	Reads the complex data from 2 streams after setting DAC daqMux

	Inherits:
		self.bay: this variable describes the bay that the card we are testing is in
	Args:
		inst: this variable is described in ReadUnknown and is not acted on by this class
		datalength: describes the size of the buffer we are setting. Because we set datalength in 32 bit
					words and we read stream data in 16 bit, we will see len(q_data) is twice datalength variable
	"""
	def __init__(self, inst, datalength, show):
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

		if show is True:
			time.sleep(0.5)
			print("Q_Data:", self.q_data[0:10], "\nI_Data:", self.i_data[0:10], "\nDAC_Data:", self.dac_data[0:10])
			time.sleep(0.5)


class FullBandResp:
	"""
	This class will take in complex data using ReadAdcData and ReadDacData
	This data is then used in welch's (signal.welch) method for estimating frequency and magnitude of the spectral density

	Args:
		band:   indicates the band on the card that we will be testing. Band can range from 0-7 if we are
				testing both cards or from 0-3 if we are only testing one card in bay 0.

	Returns:
		self.freq:  this is the frequency list we computed from running the cross power spectral density (signal.csd)
		self.resp:  this is response of csd(adc, dac) / welch(dac)
					I followed similar use of fullBandResp.m to compute self.resp in this way
	"""
	def __init__(self, band):

		noiseselectpv = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base[" + str(band) + "]:noiseSelect"
		caput(noiseselectpv, 1)
		time.sleep(0.5)

		adc_data = ReadAdcData(inst=band, datalength=2**19).adc_data
		time.sleep(0.5)

		dac_data = ReadDacData(inst=band, datalength=2**19, show=True).dac_data
		time.sleep(0.5)

		caput(noiseselectpv, 0)
		time.sleep(0.5)

		# Calculating responses

		f_x, pxx = signal.welch(x=dac_data, fs=614.4e6, return_onesided=False)
		f, pyx = signal.csd(x=adc_data, y=dac_data, fs=614.4e6, return_onesided=False)

		self.resp = []
		for index in range(len(pxx)):

			# Testing a theory of mine that pxx is equal to zero
			if pxx[index] == 0:
				time.sleep(0.5)
				print("Pxx at index", index, "is equal to zero")
				print("Length of pxx is:", len(pxx))
				time.sleep(0.5)
				break

			self.resp.append(pyx[index]/pxx[index])

		self.freq = f


if __name__ == "__main__":

	print("Testing FullBandResp with band value of 0...")
	time.sleep(1)

	fullBand = FullBandResp(band=0)
	resp = fullBand.resp
	freq = fullBand.freq

	dBresp = [20*math.log10(abs(x)) for x in resp]
	freqMhz = [x/(1e6) for x in freq]

	plt.figure(1)
	plt.title("Transfer Function")
	plt.plot(freqMhz, dBresp)
	plt.xlabel("Frequency (MHz)")
	plt.ylabel("Magnitude (dB)")
	plt.show()
else:
	print("Executed from import of ReadData")
