from GetData import ReadStreamData
from SetHardware import DaqMux
import SetHardware
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
	def __init__(self, inst, datalength, show):
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

		# For displaying graphs of I and Q data
		if show is True:
			# Plotting Q_Data vs time
			plt.title("ADC Q_Data vs Time")
			plt.plot(self.q_data)
			plt.xlabel("Time")
			plt.ylabel("Q_Data")
			plt.show()

			# Plotting I_Data vs time
			plt.title("ADC I_Data vs Time")
			plt.plot(self.i_data)
			plt.xlabel("Time")
			plt.ylabel("I_Data")
			plt.show()


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

		# For displaying graphs of I and Q data
		if show is True:
			# Plotting Q_Data vs time
			plt.title("DAC Q_Data vs Time")
			plt.plot(self.q_data)
			plt.xlabel("Time")
			plt.ylabel("Q_Data")
			plt.show()

			# Plotting I_Data vs time
			plt.title("DAC I_Data vs Time")
			plt.plot(self.i_data)
			plt.xlabel("Time")
			plt.ylabel("I_Data")
			plt.show()


class FullBandResp:
	"""
	This class will take in complex data using ReadAdcData and ReadDacData
	This data is then used in welch's (signal.welch) method for estimating frequency and magnitude of the spectral density

	Args:
		band:   indicates the band on the card that we will be testing. Band can range from 0-7 if we are
				testing both cards or from 0-3 if we are only testing one card in bay 0.

	Returns:
		self.freq:  this is the frequency list we computed from running the cross power spectral density (signal.csd)
					this list is sorted from lowest frequency to highest
		self.resp:  this is response of csd(adc, dac) / welch(dac)
					this list is sorted similarly to self.freq so that the correct resp corresponds to correct frequency
					I followed similar use of fullBandResp.m to compute self.resp in this way
	"""
	def __init__(self, band):

		# Setting all waveforms to zero
		all_waves = SetHardware.Waveform()
		all_waves.set_all_waveforms(wave_value=0)

		# Setting all attenuators to zero dB
		all_attens = SetHardware.Attenuator()
		all_attens.set_all(value_to_set=0)

		noiseselectpv = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base[" + str(band) + "]:noiseSelect"
		caput(noiseselectpv, 1)
		time.sleep(0.5)

		adc_data = ReadAdcData(inst=band, datalength=2**19, show=True).adc_data
		time.sleep(0.5)

		dac_data = ReadDacData(inst=band, datalength=2**19, show=True).dac_data
		time.sleep(0.5)

		caput(noiseselectpv, 0)
		time.sleep(0.5)

		# Calculating responses
		f_y, pyy = signal.welch(x=adc_data, nperseg=len(adc_data)//8, detrend=False, fs=614.4e6, return_onesided=False)
		f_x, pxx = signal.welch(x=dac_data, nperseg=len(adc_data)//8, detrend=False, fs=614.4e6, return_onesided=False)
		f, pyx = signal.csd(x=adc_data, y=dac_data, nperseg=len(adc_data)//8, detrend=False, fs=614.4e6, return_onesided=False)

		# Sorting ADC power spectral density data by frequency from lowest to highest
		fy_shifted, pyy_shifted = FullBandResp.freq_sort(f_y, pyy)
		self.adc_freq = fy_shifted
		self.adc_amp = pyy_shifted

		# Sorting DAC power spectral density data by frequency from lowest to highest
		fx_sorted, pxx_sorted = FullBandResp.freq_sort(f_x, pxx)
		self.dac_freq = fx_sorted
		self.dac_amp = pxx_sorted

		# Sorting cross power spectral density data by frequency from lowest to highest
		f_sorted, pyx_sorted = FullBandResp.freq_sort(f, pyx)
		self.cpsd_freq = f_sorted
		self.cpsd_amp = pyx_sorted

		self.resp = []
		for index in range(len(pxx)):

			self.resp.append(pyx_sorted[index]/pxx_sorted[index])

		self.freq = f_sorted

	@staticmethod
	def freq_sort(freq_list, amp_list):
		combined_list = []
		for index in range(len(freq_list)):
			# This creates a list of [freq, amp] lists
			# We want freq first so we can sort by frequency
			combined_list.append([freq_list[index], amp_list[index]])

		# This puts our combined list in frequency order from lowest to highest
		combined_list.sort()

		sorted_freqs = []
		sorted_amp = []
		for [frequency, amplitude] in combined_list:

			# This should give us two lists with the sorted frequencies and their corresponding amplitudes
			sorted_freqs.append(frequency)
			sorted_amp.append(amplitude)

		return sorted_freqs, sorted_amp


if __name__ == "__main__":

	def plot_psd(figure_count, title, frequency, response):
		frequency_mhz = [x/1e6 for x in frequency]
		response_db = [20*math.log10(abs(x)) for x in response]
		plt.figure(figure_count)
		plt.title(title)
		plt.plot(frequency_mhz, response_db)
		plt.xlabel("Frequency (MHz)")
		plt.ylabel("Magnitude (dB)")
		plt.show()

	print("Testing FullBandResp with band value of 0...")
	time.sleep(1)

	fullBand = FullBandResp(band=0)
	fig_count = 1

	# Plotting ADC data
	adc_title = "Adc Power Spectral Density"
	adc_freq = fullBand.adc_freq
	adc_amp = fullBand.adc_amp
	plot_psd(fig_count, adc_title, adc_freq, adc_amp)
	fig_count += 1

	# Plotting DAC data
	dac_title = "Dac Power Spectral Density"
	dac_freq = fullBand.dac_freq
	dac_amp = fullBand.dac_amp
	plot_psd(fig_count, dac_title, dac_freq, dac_amp)
	fig_count += 1

	# Plotting CPSD data
	cpsd_title = "Cross Power Spectral Density"
	cpsd_freq = fullBand.cpsd_freq
	cpsd_amp = fullBand.cpsd_amp
	plot_psd(fig_count, cpsd_title, cpsd_freq, cpsd_amp)
	fig_count += 1

	# Plotting Transfer function
	tf_title = "Transfer Function"
	tf_freq = fullBand.freq
	tf_amp = fullBand.resp
	plot_psd(fig_count, tf_title, tf_freq, tf_amp)

else:
	print("Executed from import of ReadData")
