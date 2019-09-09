from epics import caget, caput
import time
import numpy as np


class Attenuator:
	"""
	This is the parent class for UCAttenuator and DCAttenuator
	This class is used for setting the location for uc and dc.
	It is also used to set the instance for which attenuator is being set

	*Example (Setting all attenuators)*
		all_attenuators = Attenuator(atten_inst=1)
		all_attenuators.set_all()

	Args:
		atten_inst: which attenuator to be set
	Returns:
		self.location: unfinished
	"""
	# This is a list of values that an attenuator can be set to
	acceptable_values = [0, 1, 2, 4, 8, 16, 31]

	def __init__(self, atten_inst=-1):

		# Sets general location for all attenuators
		# Will need to specify UC or DC in specific attenuator Classes
		if atten_inst > 4:
			self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[1]:ATT:"
			self.inst = atten_inst - 4
		else:
			self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:"
			self.inst = atten_inst

	def set_value(self, value_to_set):
		"""
		This function sets the variable value_to_set to the attenuator location stored in self.location.
		Uses the epics function caput to set the value.

		Args:
			value_to_set: this is the value that we want to set the attenuator to
		:return:
			This function doesn't return anything and instead sets a value on the EPICs server
		"""

		# Here we make sure user entered attenuator value is a valid value to set
		if value_to_set not in Attenuator.acceptable_values:
			print("ERROR: Attenuator value invalid. Value has been set to default of 0")
			value_to_set = 0

		if self.inst == -1:
			print("ERROR: Attenuator instance not specified. Instance set to default 1")
			self.inst = 1

		if self.inst not in range(1, 5):
			print("ERROR: Attenuator instance is out of range. Instance set to default 1")
			self.inst = 1

		atten_location = self.location + "[" + str(self.inst) + "]"

		# ~~ FOR SERVER INTERFACE ~~
		caput(atten_location, value_to_set)
		time.sleep(0.1)

	def set_all(self, value_to_set):
		"""
		This function should be used to set the value of all attenuators (UC and DC).
		It uses the set_value function above to perform the bulk of this function.
		An example for how it should be used can be found in the Attenuator base class documentation.
		This function should be able to set the value of all attenuators on both hw cards

		Args:
			value_to_set: this is the value we wish to set all attenuators to
		:return:
			This function doesn't return anything, but rather sets a value on the EPICs server
		"""

		atten_types = ["UC", "DC"]
		hw_cards = [0, 1]

		for card_num in hw_cards:
			self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[" + str(card_num) + "]:ATT:"
			for converter in atten_types:
				self.location += converter
				for num in range(1, 5):
					self.inst = num
					self.set_value(value_to_set)

				# This ensures that we reset location before we append 'DC'
				self.location = self.location[:-2]


class UCAttenuator(Attenuator):
	"""
	This class inherits from Attenuator parent class
	The main reason for the existence of this class is to easily append UC to the attenuator location

	*Example (Setting one uc attenuator value to 4)*
		my_ucatten = UCAttenuator(atten_inst=1)
		my_ucatten.set_value(value_to_set=4)

	Inherits:
		self.location:  sets most of the location for where to put values for chosen attenuator
		self.inst:      set from the input argument atten_inst. Used in set_value function in Attenuator class
	"""

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "UC"


class DCAttenuator(Attenuator):
	"""
	This class inherits from Attenuator parent class
	The main reason for the existence of this class is to easily append DC to the attenuator location

	Inherits:
		self.location:  sets most of the location for where to put values for chosen attenuator
		self.inst:      set from the input argument atten_inst. Used in set_value function in Attenuator class
	"""

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "DC"


class Waveform:
	"""
	This class is used to set the values for Waveforms on the EPICs server.
	This class takes one input, waveform_inst, to specify the location of the waveform.

	*Example (Set all waveforms to 0)*
		all_waveforms = Waveform()
		all_waveforms.set_all_waveforms(wave_value=0)

	*Example (Set one waveform (inst=2) to value of 1)*
		my_wave = Waveform(waveform_inst=2)
		my_wave.set_value(wave_value=1)
	"""
	def __init__(self, waveform_inst=-1):
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base"
		self.inst = waveform_inst

	def set_value(self, wave_value):
		"""
		This function sets the value of one waveform instance to the value specified in wave_value.
		This value is set to the location defined by self.location using caput on EPICs server
		An example for how to use this function can be found in Waveform class documentation

		Args:
			wave_value: the value to set to the location in self.location
		:return:
			This function doesn't return anything, but rather it sets a value to the EPICs server using caput
		"""

		# Here we are checking that all requirements are satisfied to set a value
		if wave_value not in [0, 1]:
			print("ERROR: Wave value invalid. Value set to default 0")
			wave_value = 0

		if self.inst == -1:
			print("ERROR: Waveform instance is not defined. Set to default 0")
			self.inst = 0

		if self.inst not in range(4):
			print("ERROR: Waveform instance is out of range. Instance set to default 0")
			self.inst = 0

		wave_location = self.location + "[" + str(self.inst) + "]:waveformSelect"

		# ~~ FOR SERVER INTERFACE ~~
		caput(wave_location, wave_value)
		time.sleep(0.1)

	def set_all_waveforms(self, wave_value):
		"""
		This function sets all the waveforms to the same value specified by wave_value.
		The majority of the work done by this function is performed by the set_value function above.
		An example for how to use this function is locate in the Waveform class documentation.

		Args:
			wave_value: value to be set to all waveforms
		:return:
			This function doesn't return anything, but rather sets a value to the EPICs server using caput
		"""
		for num in range(4):
			self.inst = num
			self.set_value(wave_value)


class Buffer:
	"""
	This class is used to set the buffer size for reading data.
	It also acts as the parent class to the DaqMux class.
	The buffer can still be set by calling this class, but it is recommended to use DaqMux class.

	*Example (Setting buffer size to 2**19)*
		my_buffer = Buffer()
		my_buffer.set_buffer(size=2**19)

	Args:
		Buffer class doesn't take any arguments itself, but the function set_buffer does
	Returns:
		This init function sets locations for setting the buffer
	"""
	def __init__(self, bay):

		# Max buffer is integer value of hex(FFFFFFFF)
		# This value comes from setBufferSize.m
		self.maxBuffer = 4294967295
		self.startAddressPV = []
		self.endAddressPV = []

		if bay == 0:

			self.bufferLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:DataBufferSize'

			for num in range(4):
				self.startAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[0]:WaveformEngineBuffers:StartAddr[' + str(num) + ']')
				self.endAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[0]:WaveformEngineBuffers:EndAddr[' + str(num) + ']')

		elif bay == 1:

			self.bufferLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[1]:DataBufferSize'

			for num in range(4):
				self.startAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[1]:WaveformEngineBuffers:StartAddr[' + str(num) + ']')
				self.endAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[1]:WaveformEngineBuffers:EndAddr[' + str(num) + ']')

		else:
			print("Error in Buffer: Bay not equal to 1 or 0")

	def set_buffer(self, size=2**19):
		"""
		This function sets the buffer to the size indicated by the variable "size".
		It does this by setting the DataBufferSize variable as well as all four start and end addresses.
		Start and End addresses use 64 bit numbers, so we have to use hex to int conversion functions
		for setting these addresses

		An example for this function is shown in Buffer class documentation

		Args:
			size: this variable is used to set the size of the buffer
		:return:
			This function doesn't return anything, but rather sets values in EPICs server using caput
		"""
		bufferSize = size
		# Setting DaqMux Data buffer size

		if bufferSize > self.maxBuffer:
			print("ERROR: Buffer size entered is too large. Buffer set to max")
			bufferSize = self.maxBuffer

		# ~~ FOR SERVER INTERFACE ~~
		caput(self.bufferLocation, bufferSize)

		# Setting waveform Engine buffer size
		for index in range(4):

			# ~~ FOR SERVER INTERFACE ~~
			start_address_hex = caget(self.startAddressPV[index])

			# Added hex to int to make + 4*bufferSize work properly

			start_address_int = self.hex_to_int(start_address_hex)
			end_address_int = start_address_int + 4 * bufferSize

			# Converts end address value back to hex for caput

			end_address_hex = self.int_to_hex(end_address_int)
			caput(self.endAddressPV[index], end_address_hex)

	def show_start_end_addr(self):
		"""
		This function is used for testing purposes.
		It displays the hex values in the start and end addresses.
		I wrote this function before I knew that start and end addresses were in 64 bit hex form.

		*Example (Show the values in all start and end addresses)*
			my_buffer = Buffer()
			my_buffer.show_start_end_addr()

		:return:
			This function doesn't return anything. It access data on the EPICs server and prints
			it out in stdout
		"""
		for index in range(len(self.startAddressPV)):
			start_value = caget(self.startAddressPV[index])
			end_value = caget(self.endAddressPV[index])
			print("Start address", str(index), "value:", start_value)
			print("End address", str(index), "value:", end_value)

	def hex_to_int(self, hex_array):
		"""
		Converts a hex character string into a 64 bit integer

		Args:
		hex_array: an array of characters to be turned into an int
		return:
		int_64bit: A 64 bit integer
		"""
		int_64bit = np.int(''.join([chr(x) for x in hex_array]), 0)
		return int_64bit

	def int_to_hex(self, int_64bit):
		"""
		Converts a 64 bit integer into a hex character array

		Args:
			int_64bit: A 64 bit integer to be converted into hex
		:return:
			hex_array: A character array representing the hex format of the int
		"""
		hex_array = np.zeros(300, dtype=int)
		hex_int = hex(int_64bit)
		for j in np.arange(len(hex_int)):
			hex_array[j] = ord(hex_int[j])

		return hex_array


class DaqMux(Buffer):
	"""
	This class inherits from the Buffer class and is used to set the buffer as well as daqMux.

	*Example (Set Adc daqMux (inst=0 and datalength=2**17) and the bay=0)*
		adc_daq = DaqMux(bay=0)
		adc_daq.set_adc_daq(adcnumber=0, datalength=2**17)

	Args:
		bay: this is the bay that the card is in
	Returns:
		This init function only sets locations for future value setting
	"""
	def __init__(self, bay):
		super().__init__(bay)

		self.bay = bay

		if bay == 1:
			self.channelZeroLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[1]:InputMuxSel[0]'
			self.channelOneLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[1]:InputMuxSel[1]'
		elif bay == 0:
			self.channelZeroLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:InputMuxSel[0]'
			self.channelOneLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:InputMuxSel[1]'
		else:
			print("ERROR: Bay value unrecognized. Bay set to default 0")
			self.channelZeroLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:InputMuxSel[0]'
			self.channelOneLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:InputMuxSel[1]'

	def set_adc_daq(self, adcnumber, datalength):
		"""
		When using caput to assign Channels in variable channel0 and channel1,
		keep in mind that caput(self.channelZeroLocation, 2) corresponds to
		an output of Channel0 on the server.

		Similarly, caput(self.channelOneLocation, 3) corresponds to
		output of Channel1

		Ensures that, if instance is zero, Channel0 is selected by putting value of 2 into the PV
		Daq Mux channels are always offset by one
		"""

		daqMuxChannel0 = (adcnumber + 1) * 2
		daqMuxChannel1 = daqMuxChannel0 + 1

		my_buffer = Buffer(bay=self.bay)
		my_buffer.set_buffer(size=datalength)

		# ~~ FOR SERVER INTERFACE ~~
		caput(self.channelZeroLocation, daqMuxChannel0)
		caput(self.channelOneLocation, daqMuxChannel1)

	def set_dac_daq(self, dacnumber, datalength):
		"""
		When using caput to assign Channels in variable channel0 and channel1,
		keep in mind that caput(self.channelZeroLocation, 2) corresponds to
		an output of Channel0 on the server.

		Similarly, caput(self.channelOneLocation, 3) corresponds to
		output of Channel1

		Ensures that, if instance is zero, Channel0 is selected by putting value of 2 into the PV
		Daq Mux channels are always offset by one
		"""

		# DAC is offset by 10 compared to ADC
		daqMuxChannel0 = ((dacnumber + 1) * 2) + 10
		daqMuxChannel1 = daqMuxChannel0 + 1

		my_buffer = Buffer(bay=self.bay)
		my_buffer.set_buffer(size=datalength)

		# ~~ FOR SERVER INTERFACE ~~
		caput(self.channelZeroLocation, daqMuxChannel0)
		caput(self.channelOneLocation, daqMuxChannel1)


class SetHwTrigger:
	"""
	This class is used more like a function to set the hardware trigger.
	All it does is check the current value of the hardware trigger and switch it.
	Hardware trigger toggles back and forth from 0 to 1 when this class is called.

	*Example (change the hardware trigger value)*
		SetHwTrigger()
		# yes it is really that simple

	Args:
		None
	Returns:
		This class doesn't return anything, but rather sets the hardware trigger value on EPICs server using caput
	"""
	def __init__(self):
		hwtriggerpv = "dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:ArmHwTrigger"
		trigger_val = caget(hwtriggerpv)
		if trigger_val == 0:
			caput(hwtriggerpv, 1)
		else:
			caput(hwtriggerpv, 0)
			time.sleep(0.1)
			caput(hwtriggerpv, 1)

		time.sleep(0.1)


if __name__ == '__main__':
	print("Executed from main")
else:
	print('Executed from import of SetHardware')
