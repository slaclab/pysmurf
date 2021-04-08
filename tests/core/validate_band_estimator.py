#!/usr/bin/env python3

import argparse
import math
import numpy as np
import scipy.signal as signal

# Input arguments
parser = argparse.ArgumentParser(description='Test the SmurfProcessor Filter.')
# Filter order
parser.add_argument('--filter-order',
        type=int,
        default=4,
        dest='filter_order',
        help='Filter order')

# Filter frequency
parser.add_argument('--filter-freq',
        type=float,
        default=2*63/4000,
        dest='filter_freq',
        help='Filter order')

# Number of generated points
parser.add_argument('--num-samples',
        type=int,
        default=10000,
        dest='num_samples',
        help='Number of point to generate')

# Output directory
parser.add_argument('--out-dir',
        type=str,
        default='/tmp',
        dest='out_dir',
        help='Directory to write the output data')

# Generate plots
parser.add_argument('--plot-data',
        action='store_true',
        dest='plot_data',
        help='Plot the data')

# Do not test the rogue device
parser.add_argument('--no-rogue',
        action='store_true',
        dest='no_rogue',
        help='Do not test the rogue device.')

# Use a low pass filter before the estimator
parser.add_argument('--with-filter',
        action='store_true',
        dest='with_filter',
        help='Low pass filter the phase data before the estimator.')

class BandToneModel():
    """
    Band tone phase model.

    Model the phase of a set tones from at frequencies defined
    in "freq".

    Args
    ----
    freq : float
        List of tone frequencies, in GHz.
    jump_index : int, optional, default 10000
        Time index where the phase jump happens.
    """

    def __init__(self, freq, jump_index=10000):

        # Make sure freq is a list
        if not isinstance(freq, list):
            freq = [freq]

        # Convert to Hz
        self._freq = [f * 1e9 for f in freq]
        self._num_tones = len(freq)

        # Create a list of tone models, one for each frequency
        self._tone_models = []
        for f in self._freq:
            self._tone_models.append(
                self._TonePhaseModel(freq=f, jump_index=jump_index))

    def get_sample(self):
        """
        Generate a new phase sample set for all the band tones.

        Returns
        -------
        phase : np.ndarray
            Phase sample for each tone. The dimension will be equal to the
            number of tones.
        tau : np.ndarray
            The tau parameter used in the model, for each tone. The dimension
            will be equal to the number of tones.
        theta : np.ndarray
            the theta parameter used in the model, for each tone. The dimension
            will be equal to the number of tones.
        """
        phase = np.empty(self._num_tones)
        tau = np.empty(self._num_tones)
        theta = np.empty(self._num_tones)

        # We need to use the same tau_randn for all tones in the band
        tau_randn = np.random.randn(1)

        for m, i in zip(self._tone_models, range(self._num_tones)):
            phase[i], tau[i], theta[i] = m.get_sample(tau_randn=tau_randn)

        return phase, tau, theta

    class _TonePhaseModel():
        """
        Tone phase model.

        Model the phase of a tone at a frequency "freq" as:

            phase = tau(t) * 2*pi*freq + theta + n(t) + eta

        The phase slope "tau(t)" does a random walk over time:

            tau(t) = tau_noise_power * randon(1) + tau(t-1)

        The phase offset "theta" is constant respect to time.

        The phase noise "n(t)" is white additive noise.

        The term "eta" is a phase correction parameter, set by the user.

        Time is the index of the sample produced by the model. Every time a new
        sample is generated with the "get_sample()" method, the time index is
        incremented.

        The model also produced a "phase jump" at an specified time index,
        defined by the parameter "jump_index". The jump is modeled by changing
        tau's "tau_m" and the "theta" parameters.

        Args
        ----
        freq : float
            Tone frequency.
        jump_index : int, optional, default 10000
            Time index where the phase jump happens.
        """

        def __init__(self, freq, jump_index=10000):
            # Tone frequency
            self._freq = freq

            # Phase jump index. The
            self._jump_index = jump_index

            # Tau parameters. The "_tau_jump" will be added to tau at the jump phase
            self._tau_noise_power = 1e-13
            self._tau_jump = 1e-11
            self._prev_tau = 0

            # Theta parameters. The _theta_jump will be added to theta at the jump phase
            self._theta = -2e-3
            self._theta_jump = 3e-3

            # Noise amplitude (constant across the phase jump)
            self.noise_ampl = 2e-6

            # Sample counter
            self._sample_index = 0

            # Phase correction factor
            self._eta = 0

        def get_sample(self, tau_randn):
            """
            Returns a phase sample from the model.

            It returns a phase sample from the model.

            Args
            ----
            tau_randn : float
                The random number used to update the tau parameter.

            Returns
            -------
            phase : float
                A new phase sample from the tone model.
            tau : float
                The tau parameter used in the model.
            theta : float
                the theta parameter used in the model.
            """

            # Update Tau
            tau = self._tau_noise_power * tau_randn + self._prev_tau

            # Phase jump
            if (self._sample_index == self._jump_index):
                tau += self._tau_jump
                self._theta += self._theta_jump

            # Noise
            n = np.random.uniform(-self.noise_ampl, self.noise_ampl)

            # Phase
            phase = tau * 2 * math.pi * self._freq + self._theta + n

            # Apply the phase correction "eta"
            phase += self._eta

            # Increment the sample index
            self._sample_index += 1

            # Save tau for the next cycle
            self._prev_tau = tau

            # Return the generated phase sample
            return phase, tau, self._theta

        def set_correction(self, eta):
            """
            Set correction factor for slope and offset.

            Args:
            -----
            eta : float
                Phase correction factor.
            """
            self._eta = eta


class LowPassFilterBank():
    """
    Multi channel low pass filter.

    Args
    ----
    num_ch : int
        Number of channels.

    """

    def __init__(self, num_ch, **kwargs):
        self._num_ch = num_ch

        # Create a list of filter, one for each channel.
        self._filters = []
        for n in range(num_ch):
            self._filters.append(self._LowPassFilter(**kwargs))

    def push(self, x):
        """
        Push a new list of samples and get a new list of outputs.

        Args
        ----
        x : np.ndarray
            Input samples. The dimension of the list must be equal to the
            number of channels.

        Returns
        -------
        out : np.ndarray
            Output samples. The dimension will be equal to the number of
            channels.
        """

        if not isinstance(x, np.ndarray) or len(x) != self._num_ch:
            raise RuntimeError('Wrong input sample dimension')

        out = np.empty(self._num_ch)
        for f, xi, i in zip(self._filters, x, range(self._num_ch)):
            out[i] = f.push(xi)

        return out

    class _LowPassFilter():
        """
        Low pass filter.

        Args
        ----
        filter_order : int, optional, default 4
            Filter order.
        filter_freq : float, optional, default 2*63/4000
            Filter cutoff frequency.
        """

        def __init__(self, filter_order=4, filter_freq=2*63/4000):
            self._filter_order = filter_order
            self._filter_freq = filter_freq

            # Generate filter coefficients
            self._b, self._a = signal.butter(filter_order, filter_freq)

            # Generate the initial filter state
            self._z = signal.lfilter_zi(self._b, self._a)*[0]

        def push(self, x):
            """
            Push a new data to the filter and get a new output.

            Args
            ----
            x : float
                New data sample.

            Returns
            -------
            r : float
                New output sample.
            """
            r, self._z = signal.lfilter(self._b, self._a, [x], zi=self._z)

            return r


class BandParameterEstimator():
    """
    Band parameter estimator.

    Estimates the phase's slope (tau) and offset (theta) parameters for a band
    from the phase of two or more fixed tones.

    Args
    ----

    freq : list
        List of tone frequencies, in GHz.
    """
    def __init__(self, freq):

        # Make sure freq is a list
        if not isinstance(freq, list):
            freq = [freq]

        # Convert to Hz
        self._freq = [f * 1e9 for f in freq]

        # Make sure we have at least two tones
        self._num_tones = len(self._freq)
        if self._num_tones < 2:
            raise RuntimeError('We need at least 2 tones for the estimator')

        # Calculate the mean frequency
        self._f_mean = 2*math.pi*sum(self._freq)/len(self._freq)

        # Calculate the frequency deltas and variance
        self._f_diff = []
        self._f_var = 0
        for f in self._freq:
            d = 2*math.pi*f - self._f_mean
            self._f_diff.append(d)
            self._f_var += d**2

    def push(self, phase):
        """
        Push a new sample, and get a new estimation.

        Args
        -----
        phase : np.ndarray
            List of input phases. The dimension of the list must be equal to
            the number of tones.

        Returns
        -------
        tau : float
            Band phase slope estimation (tau).
        theta : int
            Band phase offset estimation (theta).
        """

        if not isinstance(phase, np.ndarray) or len(phase) != self._num_tones:
            raise RuntimeError('Wrong input sample dimension')

        # Estimate the band "tau" and "theta" parameters using least square
        # solution:
        #
        #     m = sum_i(x_i - x_mean)(y_i - y_mean)/sum_i(x_i - x_mean)^2
        #
        #     b = y_mean - m * x_mean
        #
        # where:
        #     x : frequency points.
        #     y : phase points.
        #     m : tau.
        #     b : theta.
        phase_mean = sum(phase)/len(phase)

        tau = 0
        for p, f_diff in zip(phase, self._f_diff):
            tau += f_diff * (p - phase_mean)
        tau /= self._f_var

        theta = phase_mean - tau * self._f_mean

        return tau, theta


if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()
    num_samples = args.num_samples
    filter_freq = args.filter_freq
    filter_order = args.filter_order
    plot_data = args.plot_data
    no_rogue = args.no_rogue
    with_filter = args.with_filter
    input_data_file = f'{args.out_dir}/input.dat'
    python_estimated_file = f'{args.out_dir}/out_python.dat'
    smurf_estimated_file = f'{args.out_dir}/out_smurf.dat'

    # Phase jump index
    # We set it at the middle of the test
    jump_index = num_samples/2

    # Tone frequencies, in GHz
    freq = [4, 4.1]

    # Arrays to hold the data
    phase = np.ndarray(shape=(num_samples, len(freq)))
    tau_model = np.ndarray(shape=(num_samples, len(freq)))
    theta_model = np.ndarray(shape=(num_samples, len(freq)))
    phase_filtered = np.ndarray(shape=(num_samples, len(freq)))
    estimation = np.ndarray(shape=(num_samples, 2))

    # Band model. We set the phase jump at the middle of the data set.
    band = BandToneModel(freq=freq, jump_index=jump_index)

    # Low pass filter bank, if enabled
    if with_filter:
        lpf = LowPassFilterBank(
            num_ch=len(freq),
            filter_order=filter_order,
            filter_freq=filter_freq)

    # Software Feedback
    estimator = BandParameterEstimator(freq=freq)

    # Generate the emulated data
    print('Generating data... ', end='')
    for i in range(num_samples):
        # Generate a phase sample, and saved the tau and theta parameters from the model
        phase[i], tau_model[i], theta_model[i] = band.get_sample()

        if with_filter:
            # Low-pass filter the data
            phase_filtered[i] = lpf.push(phase[i])
        else:
            # Do not filter the data, just copy the same generated data
            phase_filtered[i] = phase[i]

        # Run the software feedback
        estimation[i] = estimator.push(phase_filtered[i])

    print("Done")

    # Write the estimator input data into a file.
    # We convert it to "int32" as the SMuRF device will use it.
    print(f'Writing data to "{input_data_file}"... ', end='')
    np.savetxt(input_data_file, phase_filtered*(2**31), fmt='%i')
    print('Done')

    # Write the estimated data into a file.
    print(f'Writing data to "{python_estimated_file}"... ', end='')
    np.savetxt(python_estimated_file, estimation, fmt='%1.4e')
    print('Done')

    # Calculate the RMSE of python estimator, respect to the model data
    print('RMSE of the python estimator:')
    rmse_tau = np.sqrt(np.square(np.subtract(tau_model[:, 0], estimation[:, 0])).mean())
    rmse_theta = np.sqrt(np.square(np.subtract(theta_model[:, 0], estimation[:, 1])).mean())
    print(f"Tau RMSE = {rmse_tau}")
    print(f"Theta RMSE = {rmse_theta}")

    # Test the SMuRF device, if enabled
    if not no_rogue:
        import sys

        import rogue
        import rogue.interfaces.stream

        import pyrogue
        import pysmurf
        import pysmurf.core.devices
        import pysmurf.core.transmitters

        class LocalDataSource(pyrogue.Device):
            """
            Class to stream data.

            Args
            ----
            name : str
                Device name.
            description : str
                Device description.
            dataSize : int, optional, default 16
                Data size in bits to use in the frame
            """
            def __init__(self,
                 name="LocalDataSource",
                 description="Local data source",
                 dataSize=16,
                 **kwargs):
                pyrogue.Device.__init__(self, name=name, description=description, **kwargs)
                self._data_master = self._DataMaster(dataSize=dataSize)

                self.add(pyrogue.LocalVariable(
                    name='FrameCnt',
                    description='Number of sent frames',
                    mode='RO',
                    value=0,
                    localGet = self._data_master.get_frame_cnt))

                self.add(pyrogue.LocalCommand(
                    name='SendData',
                    description='Send data pass as an argument',
                    value='',
                    function=lambda arg: self._data_master.sendData(data=arg)))

            def _getStreamMaster(self):
                """
                Method called by streamConnect, streamTap and streamConnectBiDir to access master.
                """
                return self._data_master

            class _DataMaster(rogue.interfaces.stream.Master):
                """
                A Rogue master device, used to stream the data.

                Args
                ----
                dataSize : int
                    Data size in bits to use in the frame
                """
                def __init__(self, dataSize):
                    super().__init__()
                    self._frame_cnt = 0
                    self._data_size = dataSize

                def get_frame_cnt(self):
                    """
                    Get the number of sent frames
                    """
                    return self._frame_cnt

                def sendData(self, data):
                    """
                    Send a Rogue Frame. The frame contains the SMuRF header and the
                    input data points in contiguous channels. The SMuRF header will
                    be only partially filled, containing only the number of channels,
                    and a the frame counter words.

                    Args
                    ----
                    data : list
                        Input data (must be of size 'dataSize').
                    """

                    # Data size in bytes
                    data_size_bytes = int(self._data_size / 8)

                    # Request a frame to hold an SMuRF frame
                    frame = self._reqFrame(128+data_size_bytes*4096, True)

                    # Fill the frame with zeros
                    frame.write( bytearray([0]*(128+data_size_bytes*4096)), 0 )

                    # Write the number of channels
                    frame.write( bytearray((4096).to_bytes(4, sys.byteorder)), 4)

                    # Write the frame counter into the header
                    frame.write( bytearray(self._frame_cnt.to_bytes(4, sys.byteorder)), 84)

                    # Write the data into the first channel
                    index = 128 # This is the start of the data area in a SMuRF frame
                    for d in list(map(int, data.split())):
                        frame.write( bytearray(d.to_bytes(data_size_bytes, sys.byteorder, signed=True)), index)
                        index += data_size_bytes

                    # Send the frame
                    self._sendFrame(frame)

                    # Update the frame counter
                    self._frame_cnt =  self._frame_cnt + 1

        class LocalRoot(pyrogue.Root):
            """
            Local root device.

            It contains the BandPhaseFeedback device, connected to our LocalDataSource
            data source. It will read the data from the input data file generated previously,
            and send its contents through the BandPhaseEstimator device, while reading the
            estimation result variables and save then into an output file, one data set at
            a time.
            """
            def __init__(self, **kwargs):
                pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=True, **kwargs)

                # Use the LocalDataSource device as a stream data source. We will use int32 data type.
                self._streaming_stream = LocalDataSource(dataSize=32)
                self.add(self._streaming_stream)

                # Add a BandPhaseFeedback device
                self._band_phase_feedback = pysmurf.core.feedbacks.BandPhaseFeedback(name="BandPhaseFeedback", band=0)
                self.add(self._band_phase_feedback)

                # Connect the LocalDataSource data source to the BandPhaseFeedback
                pyrogue.streamConnect(self._streaming_stream, self._band_phase_feedback)

        # Send the input data through the BandPhaseFeedback device.
        # We send one frame at a time, while reading the estimator results from its
        # variables.
        print('Starting the Rogue root, and send the same data through it')
        with LocalRoot() as root:
            print('  Enabling the band phase feedback device... ', end='')
            root.BandPhaseFeedback.Disable.set(False)
            print('Done')

            print('  Setting the tone channels... ', end='')
            root.BandPhaseFeedback.toneChannels.set([0, 1])
            print('Done')

            print('  Setting the tone frequencies... ', end='')
            root.BandPhaseFeedback.toneFrequencies.set(freq)
            print('Done')

            print('  Sending the data, and writing results... ', end='')
            with open(input_data_file, 'r') as f_in, open(smurf_estimated_file, 'w') as f_out:
                for data in f_in:
                    # Send a frame with a data set from the input file
                    root.LocalDataSource.SendData.call(arg=data)

                    # Read the estimator results
                    tau = root.BandPhaseFeedback.Tau.get()
                    theta = root.BandPhaseFeedback.Theta.get()

                    # Write the results to the output file
                    f_out.write(f'{tau} {theta}\n')

            print('Done')

            # Print the number of frames sent though the estimator device
            print(f'  Number of frame sent = {root.BandPhaseFeedback.FrameCnt.get()}')

        # Load the results obtained using the BandPhaseFeedback device
        print('Reading results... ', end='')
        smurf_estimation = np.loadtxt(smurf_estimated_file, dtype='float')
        print('Done.')

        # We converted the data to 'int32' before use it on the BandPhaseFeedback device,
        # so we need to remove the 2^31 factor we introduce in the estimation results in
        # order to be able to compare it directly with the data generated with the python
        # estimator.
        smurf_estimation = smurf_estimation * (2**-31)

        # Calculate the RMSE between the python and SMuRF estimator
        rmse_tau = np.sqrt(np.square(np.subtract(estimation[:,0], smurf_estimation[:,0])).mean())
        rmse_theta = np.sqrt(np.square(np.subtract(estimation[:,1], smurf_estimation[:,1])).mean())
        print('RMSE between python and SMuRF estimators:')
        print(f'Tau = {rmse_tau}')
        print(f'Theta = {rmse_theta}')

    # Plot the data, if enable
    if plot_data:
        from matplotlib import pyplot as plt

        # Plot the tone signals
        plt.figure("Tone phase signal")
        for i in range(len(freq)):
            plt.plot(phase[:, i], label=f'Tone {i} phase')
            plt.plot(phase_filtered[:, i], label=f'Tone {i} phase (low-pass filtered)')
        plt.grid()
        plt.legend()

        # Plot the generated and estimated 'tau' parameters
        plt.figure("Phase slope (tau)")
        plt.plot(tau_model[:, 0], label="From model")
        plt.plot(estimation[:, 0], label="Estimated")
        plt.grid()
        plt.legend()

        # Plot the generated and estimated 'theta' parameters
        plt.figure("Phase offset (theta)")
        plt.plot(theta_model[:, 0], label="From model")
        plt.plot(estimation[:, 1], label="Estimated")
        plt.grid()
        plt.legend()

        plt.show()
