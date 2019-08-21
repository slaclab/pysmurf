from GetData import ReadStreamData
from SetHardware import DaqMux
from epics import caget, caput
import time
from scipy import signal
import math
import matplotlib.pyplot as plt


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


class FullBandResp:

	def __init__(self, band):

		noiseselectpv = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base[" + str(band) + "]:noiseSelect"
		caput(noiseselectpv, 1)
		time.sleep(0.5)

		adc_data = ReadAdcData(inst=band, datalength=2**19)
		time.sleep(0.5)

		dac_data = ReadDacData(inst=band, datalength=2**19)
		time.sleep(0.5)

		caput(noiseselectpv, 0)
		time.sleep(0.5)

		# Calculating responses

		f_x, pxx = signal.welch(x=dac_data, fs=614.4e6, return_onesided=False)
		f, pyx = signal.csd(x=adc_data, y=dac_data, fs=614.4e6, return_onesided=False)

		self.resp = []
		for index in range(len(pxx)):
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
