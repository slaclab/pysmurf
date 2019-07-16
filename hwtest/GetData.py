# This file will be dedicated to classes and functions that retrieve data from the server

class StreamData:

	def __init__(self, bay):

		if bay == 0:

			# Written as stream0 in matlab readStreamData.m
			# Represents the imaginary component of complex phasor data
			self.imaginary_stream = 'dans_epics:AMCc:Stream0'

			# Written as stream1 in matlab code readStreamData.m
			# This data represents the real component in complex phasor data
			self.real_stream = 'dans_epics:AMCc:Stream1'

		elif bay == 1:

			# Written as stream0 in matlab readStreamData.m
			# Represents the imaginary component of complex phasor data
			self.imaginary_stream = 'dans_epics:AMCc:Stream4'

			# Written as stream1 in matlab code readStreamData.m
			# This data represents the real component in complex phasor data
			self.real_stream = 'dans_epics:AMCc:Stream5'

		else:
			print("ERROR: Bay unrecognized. Set to default 0")
			self.imaginary_stream = 'dans_epics:AMCc:Stream0'
			self.real_stream = 'dans_epics:AMCc:Stream1'

