# from epics import caget, caput
# import time

# I think the server name is 'dans_epics'


class Attenuator:

	# This is a list of values that an attenuator can be set to
	acceptable_values = [0, 1, 2, 4, 8, 16, 31]

	def __init__(self, atten_inst=-1):

		# Sets general location for all attenuators
		# Will need to specify UC or DC in specific attenuator Classes
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT:"
		self.inst = atten_inst

	def set_value(self, value_to_set):
		# This function will set the atten_value to the self.location

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
		# caput(atten_location, value_to_set)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", atten_location)
		print("Value to set:", value_to_set)

	def set_all(self, value_to_set):
		# Sets all instances of one type of attenuator to same value

		for num in range(1, 5):
			self.inst = num
			self.set_value(value_to_set)


class UCAttenuator(Attenuator):
	# This class will inherit from the Attenuator base class

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "UC"


class DCAttenuator(Attenuator):
	# This class will inherit from the Attenuator base class

	def __init__(self, atten_inst=-1):
		super().__init__(atten_inst)
		self.location += "DC"


class Waveform:

	def __init__(self, waveform_inst=-1):
		self.location = "dans_epics:AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:Base"
		self.inst = waveform_inst

	def set_value(self, wave_value):
		# This function will set wave_value to the instance specified in init function
		# or by the set all function for all waveforms

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
		# caput(wave_location, wave_value)
		# time.sleep(0.1)

		# ~~ FOR LOCAL TESTING ~~
		print("Variable location:", wave_location)
		print("Value to set:", wave_value)

	def set_all_waveforms(self, wave_value):

		for num in range(4):
			self.inst = num
			self.set_value(wave_value)


class Buffer:

	def __init__(self):

		# Max buffer is integer value of hex(FFFFFFFF)
		# This value comes from setBufferSize.m
		self.maxBuffer = 4294967295

		self.bufferLocation = 'dans_epics:AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:DataBufferSize'
		self.startAddressPV = []
		self.endAddressPV = []

		for num in range(4):
			self.startAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[0]:WaveformEngineBuffers:StartAddr[' + str(num) + ']')
			self.endAddressPV.append('dans_epics:AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierBsa:BsaWaveformEngine[0]:WaveformEngineBuffers:EndAddr[' + str(num) + ']')

	def set_buffer(self, size=2**19):

		bufferSize = size
		# Setting DaqMux Data buffer size

		if bufferSize > self.maxBuffer:
			print("ERROR: Buffer size entered is too large. Buffer set to max")
			bufferSize = self.maxBuffer

		# ~~ FOR SERVER INTERFACE ~~
		# caput(self.bufferLocation, bufferSize)

		# ~~ FOR LOCAL TESTING ~~
		print("DaqMux Location:", self.bufferLocation)
		print("DaqMux value:", bufferSize)

		# Setting waveform Engine buffer size
		for num in range(4):

			# ~~ FOR SERVER INTERFACE ~~
			# start_address = caget(self.startAddressPV[num])
			# end_address = start_address + 4 * bufferSize
			# caput(self.endAddressPV[num], end_address)

			# ~~ FOR LOCAL TESTING ~~
			start_address = num
			end_address = start_address + 4 * bufferSize
			print("The value to be assigned to EndAdress:", end_address)


class DaqMux(Buffer):

	def __init__(self, bay):
		super().__init__()

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

	def set_daq(self, daq_type, instance, datalength):

		# When using caput to assign Channels in variable channel0 and channel1,
		# keep in mind that caput(self.channelZeroLocation, 2) corresponds to
		# an output of Channel0 on the server.
		# Similarly, caput(self.channelOneLocation, 3) corresponds to
		# output of Channel1
		# When we switch to

		daq_type = daq_type.lower()
		if daq_type == 'adc':
			# Ensures that, if instance is zero, Channel0 is selected by putting value of 2 into the PV
			# Daq Mux channels are always offset by one
			daqMuxChannel0 = (instance + 1) * 2
			daqMuxChannel1 = daqMuxChannel0 + 1
		elif daq_type == 'dac':
			# Ensures that, if instance is zero, Channel0 is selected by putting value of 12 into the PV
			# Daq Mux channels are always offset by one
			daqMuxChannel0 = ((instance + 1) * 2) + 10
			daqMuxChannel1 = daqMuxChannel0 + 1
		else:
			print("ERROR: Did not recognize daq_type entered. Setting channels to default 0 and 1")
			# Corresponds to Channel0
			daqMuxChannel0 = 2
			# Corresponds to Channel1
			daqMuxChannel1 = 3

		my_buffer = Buffer()
		my_buffer.set_buffer(size=datalength)

		# ~~ FOR SERVER INTERFACE ~~
		# caput(self.channelZeroLocation, daqMuxChannel0)
		# caput(self.channelOneLocation, daqMuxChannel1)

		# ~~ FOR LOCAL TESTING ~~
		print("Channel 0 location:", self.channelZeroLocation)
		print("Channel 0 value:", daqMuxChannel0)
		print("Channel 1 location:", self.channelOneLocation)
		print("Channel 1 value:", daqMuxChannel1)

# Variables to use for local testing
# __________________________________

# Testing UC attenuator
print("\n")
print("Testing set_attenuator function...")
my_attenuator = UCAttenuator(atten_inst=4)
my_attenuator.set_value(value_to_set=8)

# Testing set_all
print("\n")
print("Testing set_all function...")
all_attenuators = UCAttenuator()
all_attenuators.set_all(value_to_set=2)

# Testing DC attenuator
print("\n")
print("Testing DC attenuator class...")
dc_attenuator = DCAttenuator(atten_inst=1)
dc_attenuator.set_value(value_to_set=16)

# Testing waveform class
print("\n")
print("Testing Waveform class...")
my_waveform = Waveform(waveform_inst=0)
my_waveform.set_value(wave_value=1)

# Testing invalid waveform value
print("\n")
print("Testing invalid waveform value entry")
invalid_waveform = Waveform(waveform_inst=1)
invalid_waveform.set_value(wave_value=9)

# Testing invalid waveform instance
print("\n")
print("Testing invalid waveform instance...")
invalid_instance_wave = Waveform(waveform_inst=5)
invalid_instance_wave.set_value(wave_value=3)

# Testing invalid attenuator value and instance
print("\n")
print("Testing invalid instance and value for attenuator class...")
wack_attenuator = UCAttenuator(atten_inst=6)
wack_attenuator.set_value(value_to_set=40)

# Testing set_all_waveforms
print("\n")
print("Testing set_all_waveforms function...")
all_waveforms = Waveform()
all_waveforms.set_all_waveforms(wave_value=1)

# Testing Buffer
print("\n")
print("Testing set buffer...")
buffer = Buffer()
buffer.set_buffer(size=2**33)

# Testing DaqMux
print("\n")
print("Testing Daq Mux class...")
daq = DaqMux(bay=0)
daq.set_daq(daq_type='adc', instance=1, datalength=2**18)
