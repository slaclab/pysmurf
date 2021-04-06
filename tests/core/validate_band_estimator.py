#!/usr/bin/env python3

import math
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt


class BandToneModel():
    """
    Band tone phase model.

    Model the phase of a set tones from at frequencies defined
    in "freq".

    Args
    ----

    freq : float
        List of tone frequencies.
    jump_index : int, optional, default 10000
        Time index where the phase jump happens.
    """

    def __init__(self, freq, jump_index=10000):

        # Make sure freq is a list
        if not isinstance(freq, list):
            freq = [freq]

        self._freq = freq
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
            self._theta = -0.02
            self._theta_jump = 0.03

            # Noise amplitude (constant across the phase jump)
            self.noise_ampl = 2e-3

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

    def __init__(self, num_ch):

        self._num_ch = num_ch

        # Create a list of filter, one for each channel.
        self._filters = []
        for n in range(num_ch):
            self._filters.append(self._LowPassFilter())

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
        List of tone frequencies.
    """
    def __init__(self, freq):

        # Make sure freq is a list
        if not isinstance(freq, list):
            freq = [freq]
        self._freq = freq

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
    # Total number of samples.
    num_samples = 20000

    # Phase jump index
    # We set it at the middle of the test
    jump_index = num_samples/2

    # Tone frequencies
    freq = [1e9, 2e9]

    # Arrays to hold the data
    phase = np.ndarray(shape=(num_samples, len(freq)))
    tau_model = np.ndarray(shape=(num_samples, len(freq)))
    theta_model = np.ndarray(shape=(num_samples, len(freq)))
    phase_filtered = np.ndarray(shape=(num_samples, len(freq)))
    estimation = np.ndarray(shape=(num_samples, 2))

    # Band model. We set the phase jump at the middle of the data set.
    band = BandToneModel(freq=freq, jump_index=jump_index)

    # Low pass filter bank
    lpf = LowPassFilterBank(num_ch=len(freq))

    # Software Feedback
    estimator = BandParameterEstimator(freq=freq)

    for i in range(num_samples):
        # Generate a phase sample, and saved the tau and theta parameters from the model
        phase[i], tau_model[i], theta_model[i] = band.get_sample()

        # Low-pass filter the data
        phase_filtered[i] = lpf.push(phase[i])

        # Run the software feedback
        estimation[i] = estimator.push(phase_filtered[i])

    # Plot the data
    plt.figure("Tone phase signal")
    for i in range(len(freq)):
        plt.plot(phase[:, i], label=f'Tone {i} phase')
        plt.plot(phase_filtered[:, i], label=f'Tone {i} phase (low-pass filtered)')
    plt.grid()
    plt.legend()

    plt.figure("Phase slope (tau)")
    plt.plot(tau_model[:, 0], label="From model")
    plt.plot(estimation[:, 0], label="Estimated")
    plt.grid()
    plt.legend()

    plt.figure("Phase offset (theta)")
    plt.plot(theta_model[:, 0], label="From model")
    plt.plot(estimation[:, 1], label="Estimated")
    plt.grid()
    plt.legend()

    plt.show()
