
"""
Author: Hannah & Taylor
Date: 2025-09-14
honk.

"""

import numpy as np
import os
#import glob
#import sys, os
import matplotlib.pyplot as plt
# from tqdm import tqdm

from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.signal import savgol_filter, find_peaks, correlate

from VNA_import_funcs import read_file


def unwrap_for_plot(phase_array):
    "unwraps phase array for plotting"
    unwrapped_normal = np.unwrap(phase_array)
    unwrapped_shifted = np.angle(np.exp(1j*(phase_array+np.pi)))-np.pi
    if np.var(unwrapped_shifted) <= np.var(unwrapped_normal):
        return unwrapped_shifted
    else:
        return unwrapped_normal

def plot_IQ(freqs, complex_data, powers, legend_title=' ', plt_title=' ', fits=None, savepath=None):
    " plots powerscan data in IQ, amp, and phase"
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    ## IQ plot
    ax_iq = fig.add_subplot(gs[:, 0])
    ax_iq.set_title("IQ")
    ax_iq.set_xlabel("I")
    ax_iq.set_ylabel("Q")
    ax_iq.set_aspect('equal')
    ax_iq.grid(True)
    ax_iq.axvline(0, color='black')
    ax_iq.axhline(0, color='black')
    ## Amplitude plot
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_amp.set_title("Amplitude")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.grid(True)
    ## Phase plot
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_phase.set_title("Phase")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True)
    
    # Check if freqs is one dimensional 
    if freqs.ndim == 1:
        freqs = np.tile(freqs, (len(powers), 1))
        ax_phase.set_xlabel("Frequency [Hz]")
    else:
        freqs = freqs - np.mean(freqs,axis=1)[:,None]
        ax_phase.set_xlabel("$\Delta$ Frequency [Hz]")

    ## plot the things
    colors = plt.cm.rainbow(np.linspace(0, 1, len(powers)))
    for i in range(len(powers)):
        ax_iq.plot(complex_data[i].real, complex_data[i].imag, color=colors[i], markersize=2)
        ax_amp.plot(freqs[i], np.abs(complex_data[i]), color=colors[i], label=f"{powers[i]:.1f}")
        ax_phase.plot(freqs[i], unwrap_for_plot(np.angle(complex_data[i])), color=colors[i])
    ax_amp.legend(title=legend_title, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(plt_title, fontsize=18)
    if fits is not None:
        for i, fit in enumerate(fits):
            ax_iq.plot(fit.real, fit.imag, 'k--', linewidth=1)
            ax_amp.plot(freqs[i], np.abs(fit), 'k--', linewidth=1)
            ax_phase.plot(freqs[i], unwrap_for_plot(np.angle(fit)), 'k--', linewidth=1)
    #plt.tight_layout()
    if savepath:
        fname = plt_title.replace('\\', '_')
        fname = fname.replace('/', '_') + '.png'
        plt.savefig(os.path.join(savepath, fname), bbox_inches='tight')
    plt.show()

def plot_P1(freqs, powers, fits, legend_title=' ', plt_title=' ', plotpath=None):
    " plots P1 from powerscan fits"
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('P1')

    ## plot the things
    colors = plt.cm.rainbow(np.linspace(0, 1, len(powers)))
    for i in range(len(powers)):
        plt.plot(freqs,fits[i].analysis_results['P1'], color=colors[i], label=f"{powers[i]:.1f}")
    plt.legend(title=legend_title, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(plt_title, fontsize=18)

    if plotpath is not None:
        filename = plt_title if plt_title is not None else 'IQ_plot'
        if not filename.endswith('.png'):
            filename += '.png'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        plt.savefig(os.path.join(plotpath, filename))
    
    plt.show()


class res_fitter:
    def __init__(self, freqs, complex_data, fit_func, identifier=None, complex_unc=None,
                 fit_param_names=None, guess_dict=None, limits_dict=None, fix_params=None, 
                 fit_data=True, verbose=False):
        """
        Initialize the res_fitter class.
        ---
        Args:
            freqs [np.array]: Frequency data [Hz]
            complex_data [np.array]: Complex I/Q data
            fit_func [function]: Fit function that returns real-valued output (e.g. S21 real/imag concatenated)
            identifier [str, optional]: Label for the dataset
            fit_param_names [list of str]: Ordered list of parameter names for fit_func
            guess_dict [dict]: Initial guesses for fit parameters
            limits_dict [dict, optional]: Optional limits for fit parameters
            fix_params [list of str, optional]: Parameters to fix during fitting
            fit_data [bool]: If True, run fitting upon initialization
        """
        self.freqs = freqs
        self.complex_data = complex_data
        self.complex_unc = complex_unc
        self.identifier = identifier
        self.fit_func = fit_func
        self.param_names = fit_param_names
        self.guess_dict = guess_dict or {}
        self.limits_dict = limits_dict or {}
        self.fix_params = fix_params or []
        self.fit_params = None
        self.fit_param_uncs = None
        self.fit_covariance = None
        self.verbose = verbose

        if fit_data:
            self.fit()
        return

    def fine_fit(self):
        """
        Run the Minuit fit with LeastSquares.
        """
        f_proj = np.concatenate([-self.freqs, self.freqs])
        data_proj = np.concatenate([self.complex_data.real, self.complex_data.imag])
        #errors = np.full_like(data_proj, 1e-3)
        errors = np.concatenate([np.full_like(self.complex_data.real,self.complex_unc.real),
                                np.full_like(self.complex_data.imag,self.complex_unc.imag)])

        def wrapped_model(f_proj, *params):
            return self.fit_func(f_proj, *params)

        least_squares = LeastSquares(f_proj, data_proj, errors, wrapped_model)

        ## Initialize Minuit with unpacked guess_dict in correct order
        initial_values = [self.guess_dict[name] for name in self.param_names]
        m = Minuit(least_squares, *initial_values, name=self.param_names)

        ## Fix parameters if specified
        for param in self.fix_params:
            m.fixed[param] = True

        ## Apply limits if provided
        for key, lim in self.limits_dict.items():
            m.limits[key] = lim

        m.migrad()
        if not m.valid:
            print(f"Fit failed for {self.identifier}")
            return False

        m.hesse()

        self.fit_params = {name: m.values[name] for name in self.param_names}
        self.fit_param_uncs = {name: m.errors[name] for name in self.param_names}
        self.fit_covariance = m.covariance
        print(f"Fit successful for {self.identifier}")

        return True

    def fit(self):
        if not self.fine_fit():
            return False
        if self.verbose:
            print(self.fit_params)
        return True

    def plot_fit(self, savepath=None, plot_guess=False):
        """
        Plot the fit results in IQ, amplitude, and phase space
        Totally fine if fit doesn't exist. Just plots the data
        ----
        Args:
            savepath [str, optional]: If provided, save the plot to this path
            plot_guess [bool]: If True, plot the guess values as well
        """
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2)
        ax_iq = fig.add_subplot(gs[:, 0])
        ax_amp = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1])

        ax_iq.set_title("IQ"); ax_amp.set_title("Amplitude"); ax_phase.set_title("Phase")
        ax_iq.set_xlabel("I"); ax_iq.set_ylabel("Q"); ax_iq.set_aspect('equal')
        ax_phase.set_xlabel("Frequency [Hz]"); ax_phase.set_ylabel("Phase [deg]")

        ## Plot data
        ax_iq.plot(self.complex_data.real, self.complex_data.imag, label='Data', color='tab:blue')
        ax_amp.plot(self.freqs, np.abs(self.complex_data), label='Data', color='tab:blue')
        ax_phase.plot(self.freqs, unwrap_for_plot(np.angle(self.complex_data)), label='Data', color='tab:blue')

        f_proj = np.concatenate([-self.freqs, self.freqs])

        if plot_guess and self.guess_dict is not None:
            try:
                guess_vals = self.fit_func(f_proj, *[self.guess_dict[k] for k in self.param_names])
                guess_complex = guess_vals[:len(self.freqs)] + 1j * guess_vals[len(self.freqs):]

                line, = ax_iq.plot(guess_complex.real, guess_complex.imag, color='tab:orange', label='Guess')
                ax_iq.arrow(guess_complex.real[-10],
                            guess_complex.imag[-10],
                            guess_complex.real[-1]-guess_complex.real[-10],
                            guess_complex.imag[-1]-guess_complex.imag[-10],
                            shape='full',
                            length_includes_head=False,
                            color=line.get_color(),
                            width=0,
                            head_width=0.003,
                            zorder=10)
                ax_amp.plot(self.freqs, np.abs(guess_complex), color='tab:orange', label='Guess')
                ax_phase.plot(self.freqs, unwrap_for_plot(np.angle(guess_complex)), color='tab:orange', label='Guess')
            except KeyError as e:
                print(f"Missing guess value for parameter {e} — skipping guess plot.")

        if self.fit_params is not None:
            fit_vals = self.fit_func(f_proj, *[self.fit_params[k] for k in self.param_names])
            fit_complex = fit_vals[:len(self.freqs)] + 1j * fit_vals[len(self.freqs):]

            line, = ax_iq.plot(fit_complex.real, fit_complex.imag, color='crimson', label='Fit')
            ax_iq.arrow(fit_complex.real[-10],
                        fit_complex.imag[-10],
                        fit_complex.real[-1]-fit_complex.real[-10],
                        fit_complex.imag[-1]-fit_complex.imag[-10],
                        shape='full',
                        length_includes_head=False,
                        color=line.get_color(),
                        width=0,
                        head_width=0.003,
                        zorder=10)
            ax_amp.plot(self.freqs, np.abs(fit_complex), color='crimson', label='Fit')
            ax_phase.plot(self.freqs, unwrap_for_plot(np.angle(fit_complex)), color='crimson', label='Fit')

        ax_amp.legend()
        plt.suptitle(f"Fit: {self.identifier}")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath)
        plt.show()


def S21_rot_realfit_singleres(f_proj, f0, Gamma_c, Gamma_l, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi):
    """
    Using equation derived in the Taylor Hannah note
    ---------
    Args:
        f_proj: projected frequency array, should be np.concatenate([-f, f]) in Hz
        f0: resonance frequency [Hz]
        Gamma_c: cavity decay rate [Hz]
        Gamma_l: internal loss decay rate [Hz]
        Omega: Rabi frequency [rad/s]
        a_real: IQ circle scaling factor, real part
        a_imag: IQ circle scaling factor, imaginary part
        tau_real: IQ circle rotation time, real part
        tau_imag: IQ circle rotation time, imaginary part
        phi: phase offset [rad]
    ---------
    Returns:
        S21: complex-valued S21 response, with real part for negative frequencies and imaginary
    """
    f = np.abs(f_proj)
    w = 2 * np.pi * f
    dw = w - 2 * np.pi * f0
    a = a_real + 1j * a_imag
    tau = tau_real + 1j * tau_imag

    phase_factor = np.exp(1j * phi)
    prefactor = a * np.exp(-1j * dw * tau)
    Gamma_r = Gamma_c + Gamma_l
    gamma = Gamma_r/2 + Gamma_phi
    refl = -1*(Gamma_c/2/gamma) * (1 - 1j*dw/gamma) / (1 + (dw/gamma)**2 + (Omega**2/Gamma_r/gamma))
    S21 = prefactor * (1 + phase_factor*refl)
    #S21 = prefactor * (1 + refl)

    ## negative freqs are real
    real_S21 = S21.real
    real_S21[f_proj > 0] = 0

    ## positive freqs are imaginary
    imag_S21 = S21.imag
    imag_S21[f_proj < 0] = 0

    return real_S21 + imag_S21


def S21_rot_realfit_singleres_Gl0(f_proj, f0, Gamma_c, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi):
    """
    Using equation derived in the Taylor Hannah note, assuming all radiative loss is to the feedline
    ---------
    Args:
        f_proj: projected frequency array, should be np.concatenate([-f, f]) in Hz
        f0: resonance frequency [Hz]
        Gamma_c: cavity decay rate [Hz]
        Omega: Rabi frequency [rad/s]
        a_real: IQ circle scaling factor, real part
        a_imag: IQ circle scaling factor, imaginary part
        tau_real: IQ circle rotation time, real part
        tau_imag: IQ circle rotation time, imaginary part
        phi: phase offset [rad]
    ---------
    Returns:
        S21: complex-valued S21 response, with real part for negative frequencies and imaginary
    """

    return S21_rot_realfit_singleres(f_proj, f0, Gamma_c, 0, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi)

def S21_rot_realfit_doubleres(f_proj, f0, chi, Gamma_c, Gamma_l, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi, P_low):
    """
    Using equation derived in the Taylor Hannah note
    ---------
    Args:
        f_proj: projected frequency array, should be np.concatenate([-f, f]) in Hz
        f0: CENTER resonance frequency!  BETWEEN THE PARItY BANDS [Hz]
        [NEW] chi: parity band separation = (f_even - f_odd) [Hz]
        Gamma_c: cavity decay rate [Hz]
        Gamma_l: internal loss decay rate [Hz]
        Omega: Rabi frequency [rad/s]
        a_real: IQ circle scaling factor, real part
        a_imag: IQ circle scaling factor, imaginary part
        tau_real: IQ circle rotation time, real part
        tau_imag: IQ circle rotation time, imaginary part
        phi: phase offset [rad]
        [NEW] P_low : probability of left parity band
    ---------
    Returns:
        S21: complex-valued S21 response, with real part for negative frequencies and imaginary
    """
    
    f_lo = f0 - chi/2
    f_hi = f0 + chi/2

    return S21_rot_realfit_singleres(f_proj, f_lo, Gamma_c, Gamma_l, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi)*P_low + \
           S21_rot_realfit_singleres(f_proj, f_hi, Gamma_c, Gamma_l, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi)*(1-P_low)



def S21_rot_realfit_doubleres_Gl0(f_proj, f0, chi, Gamma_c, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi, P_low):
    """
    Using equation derived in the Taylor Hannah note
    ---------
    Args:
        f_proj: projected frequency array, should be np.concatenate([-f, f]) in Hz
        f0: CENTER resonance frequency!  BETWEEN THE PARItY BANDS [Hz]
        [NEW] chi: parity band separation = (f_even - f_odd) [Hz]
        Gamma_c: cavity decay rate [rad/s]
        Omega: Rabi frequency [rad/s]
        a_real: IQ circle scaling factor, real part
        a_imag: IQ circle scaling factor, imaginary part
        tau_real: IQ circle rotation time, real part
        tau_imag: IQ circle rotation time, imaginary part
        phi: phase offset [rad]
        [NEW] P_low : probability of left parity band
    ---------
    Returns:
        S21: complex-valued S21 response, with real part for negative frequencies and imaginary
    """

    return S21_rot_realfit_doubleres(f_proj, f0, chi, Gamma_c, 0, Gamma_phi, Omega, a_real, a_imag, tau_real, tau_imag, phi, P_low)

def calculate_SQUAT_dependent_params(result_dictionary, uncertainties=None, covariance=None):

    Gamma_r = result_dictionary['Gamma_c'] + result_dictionary.get('Gamma_l', 0)
    gamma = Gamma_r/2 + result_dictionary['Gamma_phi']
    T1 = 1/Gamma_r
    T2 = 1/gamma
    if result_dictionary['Gamma_phi'] == 0:
        Tphi = 1e30
    else:
        Tphi = 1/result_dictionary['Gamma_phi']

    result_dictionary.update({'Gamma_r': Gamma_r, 'gamma': gamma, 'T1': T1, 'T2': T2, 'Tphi': Tphi})
    
    if covariance is not None:
        if 'Gamma_l' not in result_dictionary:
            covariance_Gamma_l = {}
            covariance_Gamma_l = {key: 0 for key in result_dictionary}
            covariance_Gamma_l['Gamma_l'] = 0
        else:
            covariance_Gamma_l = covariance['Gamma_l']
        Gamma_r_unc = np.sqrt(covariance['Gamma_c']['Gamma_c'] + covariance_Gamma_l['Gamma_l'] + 2*covariance_Gamma_l['Gamma_c'])
        gamma_unc = np.sqrt(0.25*covariance['Gamma_c']['Gamma_c'] + 0.25*covariance_Gamma_l['Gamma_l'] + covariance['Gamma_phi']['Gamma_phi'] + 0.5*covariance_Gamma_l['Gamma_c'] + covariance['Gamma_phi']['Gamma_c'] + covariance_Gamma_l['Gamma_phi'])
        T1_unc = Gamma_r_unc / (Gamma_r**2)
        T2_unc = gamma_unc / (gamma**2)
        Tphi_unc = covariance['Gamma_phi']['Gamma_phi']**0.5 / (result_dictionary['Gamma_phi']**2)

        uncertainties.update({'Gamma_r': Gamma_r_unc, 'gamma': gamma_unc, 'T1': T1_unc, 'T2': T2_unc, 'Tphi': Tphi_unc})

    return result_dictionary, uncertainties


def print_fit_results_either(result_dictionary, covariance=None):
    for key in result_dictionary:
            if key in ['f0']:
                print(f'  {key}: {result_dictionary[key]/1e9:.6f} GHz')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5/1e9:.6f} GHz')
            elif key in ['chi', 'Gamma_c', 'Gamma_l', 'Gamma_phi']:
                print(f'  {key}: {result_dictionary[key]/1e6:.3f} MHz')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5/1e6:.3f} MHz')
            elif key in ['Omega']:
                print(f'  {key}: {covariance[key][key]**0.5/(2*np.pi)/1e6:.3f} MHz')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5/(2*np.pi)/1e6:.3f} MHz')
            elif key in ['tau_real', 'tau_imag']:
                print(f'  {key}: {result_dictionary[key]*1e9:.3f} ns')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5*1e9:.3f} ns')
            elif key in ['phi']:
                print(f'  {key}: {result_dictionary[key]:.3f} rad')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5:.3f} rad')
            elif key in ['a_real', 'a_imag', 'P_low']:
                print(f'  {key}: {result_dictionary[key]:.3f}')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5:.3f}')
            else:
                print(f'  {key}: {result_dictionary[key]:.3f} ?')
                if covariance is not None:
                    print(f'    {key} unc: {covariance[key][key]**0.5:.3f} ?')
    print('-'*8, '\nDependent parameters:')
    Gamma_r = result_dictionary['Gamma_c'] + result_dictionary.get('Gamma_l', 0)
    gamma = Gamma_r/2 + result_dictionary['Gamma_phi']
    
    if covariance is not None and 'Gamma_l' not in result_dictionary:
        covariance_Gamma_l = {}
        covariance_Gamma_l = {key: 0 for key in result_dictionary}
        covariance_Gamma_l['Gamma_l'] = 0
    else:
        covariance_Gamma_l = covariance['Gamma_l']

    print(f'    Gamma_r: {Gamma_r/1e6:.3f} MHz')
    if covariance is not None:
        Gamma_r_unc = np.sqrt(covariance['Gamma_c']['Gamma_c'] + covariance_Gamma_l['Gamma_l'] + 2*covariance_Gamma_l['Gamma_c'])
        print(f'      Gamma_r unc: {Gamma_r_unc/1e6:.3f} MHz')
    print(f'    gamma: {gamma/1e6:.3f} MHz')
    if covariance is not None:
        gamma_unc = np.sqrt(0.25*covariance['Gamma_c']['Gamma_c'] + 0.25*covariance_Gamma_l['Gamma_l'] + covariance['Gamma_phi']['Gamma_phi'] + 0.5*covariance_Gamma_l['Gamma_c'] + covariance['Gamma_phi']['Gamma_c'] + covariance_Gamma_l['Gamma_phi'])
        print(f'      gamma unc: {gamma_unc/1e6:.3f} MHz')
   
    print(f'    T_1: {1/Gamma_r*1e6:.3f} us')
    if covariance is not None:
        T1_unc = Gamma_r_unc / (Gamma_r**2)
        print(f'      T_1 unc: {T1_unc * 1e6:.3f} us')
    print(f'    T_2: {1/gamma*1e6:.3f} us')
    if covariance is not None:
        T2_unc = gamma_unc / (gamma**2)
        print(f'      T_2 unc: {T2_unc * 1e6:.3f} us')
    print(f'    T_phi: {1/result_dictionary["Gamma_phi"]*1e6:.3f} us')
    if covariance is not None: 
        Tphi_unc = covariance['Gamma_phi']['Gamma_phi']**0.5 / (result_dictionary['Gamma_phi']**2)
        print(f'      T_phi unc: {Tphi_unc * 1e6:.3f} us')



class squat_S21_resfit():
    """Fit Parameter Definitions:
    ---------
    f0: resonance frequency (or central frequency between bands)
    chi: parity band separation = (f_even - f_odd) [Hz]
    Gamma_c: cavity decay rate [Hz]
    Gamma_phi: pure dephasing rate [Hz]
    Gamma_l: internal loss rate [Hz]
    Omega: Rabi frequency [rad/s]
    a_real: IQ circle scaling factor, real part
    a_imag: IQ circle scaling factor, imaginary part
    tau_real: IQ circle rotation time, real part
    tau_imag: IQ circle rotation time, imaginary part
    phi: phase offset [rad]
    (if double res) chi: parity band separation = (f_even - f_odd) [Hz]
    (if double res) P_low : probability of left parity band
    """
    ################################################################################

    def __init__(self, freqs, complex_data, identifier=' ',
                 Gl0_guess_dict=None, Gl0_limits_dict=None,
                 NonGl0_guess_dict=None, NonGl0_limits_dict=None, 
                 force_singleres=False, force_doubleres=False,
                 ):
        """
        ---
        Args:
            freqs [np.array]: Frequency data [Hz]
            complex_data [np.array]: Complex I/Q data
            identifier [str, optional]: Label for the dataset
            Gl0_guess_dict [dict, optional]: Overwrite guesses for the coarse Gl=0 fit
            Gl0_limits_dict [dict, optional]: Overwrite limits for the coarse Gl=0 fit
            NonGl0_guess_dict [dict, optional]: Overwrite guesses for the fine NonGl=0 fit
            NonGl0_limits_dict [dict, optional]: Overwrite limits for the fine NonGl=0 fit
            force_singleres [bool, optional]: Force fit to single resonance
            force_doubleres [bool, optional]: Force fit to double resonance
        """
        self.data = {
            'freqs' : freqs,
            'complex_data' : complex_data,
            'identifier' : identifier
        }
        self.flags = {
            'force_singleres' : force_singleres,
            'force_doubleres' : force_doubleres
        }
        self.res_num = {}            ## Related to guessing number of resonances
        self.Gl0_fit_results = {}    ## Results from the coarse Gl=0 fit
        self.NonGl0_fit_results = {} ## Results from the fine NonGl=0 fit
        self.analysis_results = {}   ## Post-processing of fit results

        self.ovrde_gvals_Gl0 = Gl0_guess_dict
        self.ovrde_limits_Gl0 = Gl0_limits_dict
        self.ovrde_gvals_NonGl0 = NonGl0_guess_dict
        self.ovrde_limits_NonGl0 = NonGl0_limits_dict

        return
    

    
    def guess_Gl0(self, freqs=None, data=None, plot=False):
        """Guess values for the Gl=0 fit
           If no data is supplied, use the data from the object
           ---------
           Args:
               freqs [np.array, optional]: Frequency data [Hz]
               data [np.array, optional]: Complex I/Q data
               plot [bool, optional]: debug plot
            Outputs:
                self.gvals_Gl0 [dict]: Dictionary of guess params saved to class             
        """
        self.gvals_Gl0 = {}
        if freqs is None: freqs = self.data['freqs']
        if data is None: data = self.data['complex_data']

        ## Pull the edges of the trace, fit to a line
        edge_data_f = np.hstack([freqs[:int(len(freqs)/10)], freqs[-int(len(freqs)/10):]])
        edge_data_z = np.hstack([data[:int(len(data)/10)], data[-int(len(data)/10):]])
        self.data['complex_unc'] = np.std(edge_data_z.real)+1j*np.std(edge_data_z.imag)
        edge_data_z_low = data[:int(len(data)/10)]
        edge_data_z_high = data[-int(len(data)/10):]
        #edge_data_phase = np.hstack([np.unwrap(np.angle(data))[:int(len(data)/10)], np.unwrap(np.angle(data))[-int(len(data)/10):]])
        linfit_edges_real = np.polyfit(edge_data_f, edge_data_z.real, 1)
        linfit_edges_imag = np.polyfit(edge_data_f, edge_data_z.imag, 1)

        ## Guess for 'A' is center location of the edge trace 
        self.gvals_Gl0['a_real'] = np.mean(edge_data_z.real)
        self.gvals_Gl0['a_imag'] = np.mean(edge_data_z.imag)

        ## Guess for 'phi' is the zero because this works better than any of my guesses
        self.gvals_Gl0['phi'] = 0 #np.angle(self.gvals_Gl0['a_real'] + 1j*self.gvals_Gl0['a_imag'])

        ## Guess for 'tau' is slope of the line
        self.gvals_Gl0['tau_real'] = -1*linfit_edges_real[0]/2/np.pi
        self.gvals_Gl0['tau_imag'] = -1*linfit_edges_imag[0]/2/np.pi

        ## TO-DO - make better guess for Gamma_c
        df = np.mean(freqs[-int(len(freqs)/10):])-np.mean(freqs[:int(len(freqs)/10)])
        dS21 = np.mean(data[-int(len(data)/10):])-np.mean(data[:int(len(data)/10)])

        self.gvals_Gl0['Gamma_c'] = abs(-np.pi*df*(dS21.imag/abs(np.mean(edge_data_z))))
        self.gvals_Gl0['Gamma_phi'] = self.gvals_Gl0['Gamma_c'] / 100

        ## [HM, 9/16/2025] adding this back bc if the taylor function fails, we need a backup
        self.gvals_Gl0['f0'] = np.mean(freqs)

        ## Find ratio of semimajor and semiminor diameters
        ## Use this ratio to scale Omega w.r.t Gamma_c. This works surprisingly well.
        X = np.column_stack([data.real, data.imag])
        u, s, vh = np.linalg.svd(X - X.mean(axis=0))
        diameter_ratio = s[0]/s[-1]

        # Taylor did the math and this is how Omega and Gamma_c are related to the diameter ratio
        self.gvals_Gl0['Omega'] = self.gvals_Gl0['Gamma_c']*np.sqrt(0.5*((diameter_ratio**2)-1))

        #guess_vals = S21_rot_realfit_singleres_Gl0(np.concatenate([-freqs, freqs]), self.gvals_Gl0['f0'], self.gvals_Gl0['Gamma_c'], self.gvals_Gl0['Gamma_phi'], self.gvals_Gl0['Omega'], self.gvals_Gl0['a_real'], self.gvals_Gl0['a_imag'], 0, 0, 0)
        guess_vals = S21_rot_realfit_singleres_Gl0(np.concatenate([-freqs, freqs]),
                                                   self.gvals_Gl0['f0'],
                                                   2*self.gvals_Gl0['Gamma_c'],
                                                   0*self.gvals_Gl0['Gamma_phi'],
                                                   self.gvals_Gl0['Omega'],
                                                   self.gvals_Gl0['a_real'],
                                                   self.gvals_Gl0['a_imag'],
                                                   0,
                                                   0,
                                                   0)
        guess_complex = guess_vals[:len(freqs)] + 1j * guess_vals[len(freqs):]
        #MF_abs = (correlate(abs(data)-abs(np.mean(edge_data_z)), abs(guess_complex[::1])-abs(np.mean(edge_data_z)), mode='same'))
        #MF_phase = (correlate(np.unwrap(np.angle(data))-np.mean(edge_data_phase), np.unwrap(np.angle(guess_complex[::1]))-np.mean(edge_data_phase), mode='same'))
        
        data_complex_padded = np.hstack([np.mean(edge_data_z_low)*np.ones(len(data)),data,np.mean(edge_data_z_high)*np.ones(len(data))])
        MF_complex = abs(correlate(data_complex_padded-np.mean(edge_data_z), guess_complex[::1]-np.mean(edge_data_z), mode='same'))
        MF_complex = MF_complex[len(data):-len(data)]
        MF_complex = MF_complex / np.max(MF_complex)

        prominence = 0.05
        peaks, _ = find_peaks(MF_complex, prominence=prominence)
        dips, _  = find_peaks(-MF_complex, prominence=prominence)
        while len(peaks)>2:
            prominence+=0.05
            peaks, _ = find_peaks(MF_complex, prominence=prominence)
            dips, _  = find_peaks(-MF_complex, prominence=prominence)
        if plot: print(f'Found {len(peaks)} peaks and {len(dips)} dips in the data')
        
        if len(peaks) >= 2 and len(dips) >= 1:
            self.res_num['guess_doubleres'] = True
            #self.gvals_Gl0['Gamma_c'] *= 1.5
            #self.gvals_Gl0['Gamma_phi'] *= 0.5
            #self.gvals_Gl0['Omega'] = 0.5*np.sqrt(0.5*self.gvals_Gl0['Omega']**2 - self.gvals_Gl0['Gamma_c']**2)
            #self.gvals_Gl0['Omega'] *= 0.5
            self.gvals_Gl0['P_low'] = MF_complex[peaks[0]]/(MF_complex[peaks[0]]+MF_complex[peaks[1]])
        else: self.res_num['guess_doubleres'] = False

        self.res_num['peak_locs'] = self.data['freqs'][peaks]

        if plot:
            plt.figure(figsize=(6,4),dpi=120)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.plot(abs(data))
            ax1.plot(abs(guess_complex))
            ax2.plot(MF_complex,c='k')
            for peak in peaks:
                ax2.axvline(peak, color='green', linestyle='--')
            for dip in dips:
                ax2.axvline(dip, color='red', linestyle='--')
            plt.show()

        return



    def fit_Gl0(self, verbose=True, plot=False):
        """Initial fit, assuming no internal loss (Gamma_l=0)
        If data looks like two resonances, fit to double resonance model
        Override this with force_singleres or force_doubleres provided in init
        ---------
        Args:
            verbose [bool, optional]: If True, spit out debug results
            plot [bool, optional]: If True, plot the fit results
        Outputs:
            self.Gl0_fit_results [dict]: Dictionary of fit results saved to class
        """

        ## Initialize guesses
        self.guess_Gl0(plot=verbose)

         ## ---------------------  SINGLE RES ------------------------------
        if (not self.res_num['guess_doubleres'] or self.flags['force_singleres']) and not self.flags['force_doubleres']:
            if verbose: print('-'*12, '\nFitting Gl=0 single res')
            self.res_num['fitted_to_double_res'] = False

            ## Initialize limits
            limits_dict = {
                    'Gamma_c': (0, 1e9),
                    'Gamma_phi': (0, 1e9),
                    'Omega': (0, 1e9)}

            ## If available, update the f0 guess with the identified peak
            if len(self.res_num['peak_locs']) != 0:
                self.gvals_Gl0['f0'] = self.res_num['peak_locs'][0]
            else:
                print('No peak found in data! Using center of trace as f0 guess.')

            if self.ovrde_gvals_Gl0 is not None:
                for key in self.ovrde_gvals_Gl0:
                    self.gvals_Gl0[key] = self.ovrde_gvals_Gl0[key]
            if self.ovrde_limits_Gl0 is not None:
                for key in self.ovrde_limits_Gl0:
                    limits_dict[key] = self.ovrde_limits_Gl0[key]

            ## Run the fit
            res_fit_Gl0 = res_fitter(
                self.data['freqs'],
                self.data['complex_data'],
                fit_func = S21_rot_realfit_singleres_Gl0,
                identifier='Gl=0 fit singleres; ' + self.data['identifier'],
                complex_unc=self.data['complex_unc'],
                fit_param_names=['f0', 'Gamma_c', 'Gamma_phi', 'Omega', 
                                'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi'], 
                guess_dict = self.gvals_Gl0,
                limits_dict = limits_dict,
                fit_data=True)

        ## --------------------- DOUBLE RES ------------------------------
        else:
            if verbose: print('-'*12, '\nFitting Gl=0 double res')
            self.res_num['fitted_to_double_res'] = True

            ## Initialize limits
            limits_dict = {
                    'Gamma_c': (0, 1e9),
                    'Gamma_phi': (0, 1e9),
                    'Omega': (0, 1e9),
                    'chi' : (0, (self.data['freqs'][-1] - self.data['freqs'][0])/2),
                    'P_low' : (0, 1)}

            ## Add guesses
            if 'P_low' not in self.gvals_Gl0:
                self.gvals_Gl0['P_low'] = 0.5
            if 'chi' not in self.gvals_Gl0:
                if len(self.res_num['peak_locs']) == 2:
                    self.gvals_Gl0['chi'] = np.abs(self.res_num['peak_locs'][-1] - self.res_num['peak_locs'][0])
                    self.gvals_Gl0['f0'] = np.mean(self.res_num['peak_locs'])
                elif len(self.res_num['peak_locs']) == 1:
                    self.gvals_Gl0['chi'] = 0
                    self.gvals_Gl0['f0'] = self.res_num['peak_locs'][0]
                else:
                    self.gvals_Gl0['chi'] = (self.data['freqs'][-1] - self.data['freqs'][0]) / 8
                    print('No peak found in data! Using center of trace as f0 guess. and 1/8 of trace width as chi')

                    

            if self.ovrde_gvals_Gl0 is not None:
                for key in self.ovrde_gvals_Gl0:
                    self.gvals_Gl0[key] = self.ovrde_gvals_Gl0[key]
            if self.ovrde_limits_Gl0 is not None:
                for key in self.ovrde_limits_Gl0:
                    limits_dict[key] = self.ovrde_limits_Gl0[key]

            res_fit_Gl0 = res_fitter(
                self.data['freqs'],
                self.data['complex_data'],
                fit_func = S21_rot_realfit_doubleres_Gl0,
                identifier='Gl=0 fit doubleres; ' + self.data['identifier'],
                complex_unc=self.data['complex_unc'],
                fit_param_names=['f0', 'chi', 'Gamma_c', 'Gamma_phi', 'Omega', 
                                'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi', 'P_low'], 
                guess_dict = self.gvals_Gl0,
                limits_dict = limits_dict,
                fit_data=True)
            
        ## --------------------- END FITTING ------------------------------
        if verbose and res_fit_Gl0.fit_params is not None:
            print_fit_results_either(res_fit_Gl0.fit_params, res_fit_Gl0.fit_covariance)
        if plot and res_fit_Gl0.fit_params is not None: res_fit_Gl0.plot_fit(plot_guess=True)
        
        self.Gl0_fit_results = res_fit_Gl0.fit_params
        self.Gl0_fit_result_uncs = res_fit_Gl0.fit_param_uncs
        self.Gl0_fit_covariance = res_fit_Gl0.fit_covariance

        if verbose: print('Gl=0 fit results:', self.Gl0_fit_results, '\n', '-'*12)
        
        return self.Gl0_fit_results
    

    def fit_NonGl0(self, fix_params=None, verbose=True, plot=True):
        """Fine fit, allowing internal loss (Gamma_l > 0)
        Uses results from the Gl=0 fit as initial guesses
        ---------
        Args:
            verbose [bool, optional]: If True, spit out debug results
            fix_params [list, optional]: List of parameters to fix to their Gl=0 fit values
            plot [bool, optional]: If True, plot the fit results
        Outputs:
            self.NonGl0_fit_results [dict]: Dictionary of fit results saved to class
        """
        if not self.Gl0_fit_results:
            print('Error! No Gl=0 fit results found. Run fit_Gl0() first.')
            return

        ## Initialize guesses from Gl=0 fit
        self.gvals_NonGl0 = self.Gl0_fit_results.copy()
        if self.ovrde_gvals_NonGl0 is not None:
            for key in self.ovrde_gvals_NonGl0:
                self.gvals_NonGl0[key] = self.ovrde_gvals_NonGl0[key]
        if 'Gamma_l' not in self.gvals_NonGl0:
            self.gvals_NonGl0['Gamma_l'] = self.gvals_NonGl0['Gamma_c'] / 100

        ## ---------------------  SINGLE RES ------------------------------
        if not self.res_num['fitted_to_double_res']:
            if verbose: print('-'*12, '\nFitting NonGl=0 single res')

            ## Initialize limits, bounding Gamma_c and Gamma_phi to within 20% of Gl=0 fit results
            limits_dict = {
                    'Gamma_c': (0.8*self.Gl0_fit_results['Gamma_c'], 1.2*self.Gl0_fit_results['Gamma_c']),
                    'Gamma_l': (0, self.Gl0_fit_results['Gamma_c']/2),
                    'Gamma_phi': (0.8*self.Gl0_fit_results['Gamma_phi'], 1.2*self.Gl0_fit_results['Gamma_phi']),
                    'Omega': (0.8*self.Gl0_fit_results['Omega'], 1.2*self.Gl0_fit_results['Omega'])}
            if self.ovrde_limits_NonGl0 is not None:
                for key in self.ovrde_limits_NonGl0:
                    limits_dict[key] = self.ovrde_limits_NonGl0[key]

            ## Run the fit
            res_fit_NonGl0 = res_fitter(
                self.data['freqs'],
                self.data['complex_data'],
                fit_func = S21_rot_realfit_singleres,
                identifier='NonGl=0 fit singleres; ' + self.data['identifier'],
                complex_unc=self.data['complex_unc'],
                fit_param_names=['f0', 'Gamma_c', 'Gamma_l', 'Gamma_phi', 'Omega', 
                                'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi'], 
                guess_dict = self.gvals_NonGl0,
                limits_dict = limits_dict,
                fix_params = fix_params,
                fit_data=True)
        
        ## --------------------- DOUBLE RES ------------------------------
        else:
            if verbose: print('-'*12, '\nFitting NonGl=0 double res')

            ## Initialize limits, bounding Gamma_c and Gamma_phi to within 20% of Gl=0 fit results
            limits_dict = {
                    'Gamma_c': (0.8*self.Gl0_fit_results['Gamma_c'], 1.2*self.Gl0_fit_results['Gamma_c']),
                    'Gamma_l': (0, self.Gl0_fit_results['Gamma_c']/2),
                    'Gamma_phi': (0.8*self.Gl0_fit_results['Gamma_phi'], 1.2*self.Gl0_fit_results['Gamma_phi']),
                    'Omega': (0.8*self.Gl0_fit_results['Omega'], 1.2*self.Gl0_fit_results['Omega']),
                    'chi' : (0.8*self.Gl0_fit_results['chi'], 1.2*self.Gl0_fit_results['chi']),
                    'P_low' : (0, 1)}
            if self.ovrde_limits_NonGl0 is not None:
                for key in self.ovrde_limits_NonGl0:
                    limits_dict[key] = self.ovrde_limits_NonGl0[key]

            ## Run the fit
            res_fit_NonGl0 = res_fitter(
                self.data['freqs'],
                self.data['complex_data'],
                fit_func = S21_rot_realfit_doubleres,
                identifier='NonGl=0 fit doubleres; ' + self.data['identifier'],
                complex_unc=self.data['complex_unc'],
                fit_param_names=['f0', 'chi', 'Gamma_c', 'Gamma_l', 'Gamma_phi', 'Omega', 
                                'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi', 'P_low'], 
                guess_dict = self.gvals_NonGl0,
                limits_dict = limits_dict,
                fix_params = fix_params,
                fit_data=True)
            
        ## --------------------- END FITTING ------------------------------
        if verbose and res_fit_NonGl0.fit_params is not None:
            print_fit_results_either(res_fit_NonGl0.fit_params, res_fit_NonGl0.fit_covariance)
        
        if plot and res_fit_NonGl0.fit_params is not None: res_fit_NonGl0.plot_fit(plot_guess=True)
        
        self.NonGl0_fit_results = res_fit_NonGl0.fit_params
        self.NonGl0_fit_result_uncs = res_fit_NonGl0.fit_param_uncs
        self.NonGl0_fit_covariance = res_fit_NonGl0.fit_covariance

        if verbose: print('NonGl=0 fit results:', self.NonGl0_fit_results, '\n', '-'*12)
        
        return self.NonGl0_fit_results
    


    def fit(self, fix_params=None, plot_Gl0=True, plot_NonGl0=True, verbose=False):
        """Run both the Gl=0 and NonGl=0 fits in sequence
        ---------
        Args:
            verbose [bool, optional]: If True, spit out debug results
            fix_params [list, optional]: List of parameters to fix to their Gl=0 fit values
            plot [bool, optional]: If True, plot the fit results
        Outputs:
            self.Gl0_fit_results [dict]: Dictionary of Gl=0 fit results saved to class  
        """
        self.fit_Gl0(verbose=verbose, plot=plot_Gl0)
        if not self.Gl0_fit_results:
            print('Error! The Gl=0 fit failed.  Cannot run NonGl=0 fit.')
            return None
        self.fit_NonGl0(fix_params=fix_params, verbose=verbose, plot=plot_NonGl0)
        return self.NonGl0_fit_results
    


    def plot_fit(self, plot_guess=True):
        """Plot the NonGl=0 fit results already stored in the class
        Args:
            plot_guess [bool, optional]: If True, plot the initial guess as well
        """
        if not self.NonGl0_fit_results:
            print('Error! No NonGl=0 fit results found. Run fit_NonGl0() first.')
            return
        res_fit_NonGl0 = res_fitter(
            self.data['freqs'],
            self.data['complex_data'],
            fit_func = S21_rot_realfit_doubleres if self.res_num['fitted_to_double_res'] else S21_rot_realfit_singleres,
            identifier='NonGl=0 fit doubleres; ' + self.data['identifier'] if self.res_num['fitted_to_double_res'] else 'NonGl=0 fit singleres; ' + self.data['identifier'],
            complex_unc=self.data['complex_unc'],
            fit_param_names=['f0', 'chi', 'Gamma_c', 'Gamma_l', 'Gamma_phi', 'Omega', 
                            'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi', 'P_low'] if self.res_num['fitted_to_double_res'] else ['f0', 'Gamma_c', 'Gamma_l', 'Gamma_phi', 'Omega', 
                            'a_real', 'a_imag', 'tau_real', 'tau_imag', 'phi'], 
            guess_dict = self.gvals_NonGl0,
            limits_dict = None,
            fix_params = None,
            fit_data=False)
        res_fit_NonGl0.fit_params = self.NonGl0_fit_results
        res_fit_NonGl0.plot_fit(plot_guess=plot_guess)
        return
    

    def print_fit_results(self):
        """Print the results of the Gl=0 and NonGl=0 fits in a readable format
        """
        if self.Gl0_fit_results:
            print('-'*12, '\nGl=0 fit results:')
            print_fit_results_either(self.Gl0_fit_results, self.Gl0_fit_covariance)
        else:
            print('No Gl=0 fit results found.')

        if self.NonGl0_fit_results:
            print('-'*12, '\nNonGl=0 fit results:')
            print_fit_results_either(self.NonGl0_fit_results, self.NonGl0_fit_covariance)
        else:
            print('No NonGl=0 fit results found.')
        print('-'*12)
        return
    

    def calc_sigz(self, use_NonGl0_result=True, plot=True, freq_choice=None):
        """Calculate the qubit sigma_z values from the fit results
        Args:
            use_NonGl0_result [bool, optional]: If True, use the NonGl=0 fit results. If False, use the Gl=0 fit results
            plot [bool, optional]: If True, plot the sigz results
        Outputs:
            self.analysis_results['sigz_vals'] [np.array]: Array of sigma_z values at each frequency point
        """
        sigz_val_choice = None
        
        if use_NonGl0_result:
            if len(self.NonGl0_fit_results.keys()) == 0:
                print("Error: No Non-Gl0 fit results to calculate sigz from.")
                return None
            fit_params = self.NonGl0_fit_results
            gamma = fit_params['Gamma_c']/2 + fit_params['Gamma_l']/2 + fit_params['Gamma_phi']
            Gamma_r = fit_params['Gamma_c'] + fit_params['Gamma_l']
        else:
            if self.Gl0_fit_results is None:
                print("Error: No Gl0 fit results to calculate sigz from.")
                return None
            fit_params = self.Gl0_fit_results
            Gamma_r = fit_params['Gamma_c']
        gamma = Gamma_r/2 + fit_params['Gamma_phi']
        Omega = fit_params['Omega']

        sigz_vals = np.zeros(len(self.data['freqs']))
        if self.res_num['guess_doubleres'] or self.flags['force_doubleres']:
            for idx, f in enumerate(self.data['freqs']):
                Delta_low = 2*np.pi*(f - fit_params['f0'] +0.5*fit_params['chi'])
                Delta_high = 2*np.pi*(f - fit_params['f0'] -0.5*fit_params['chi'])
                num_low = 1 + (Delta_low/gamma)**2
                denom_low = 1 + (Delta_low/gamma)**2 + (Omega**2/gamma/Gamma_r)
                num_high = 1 + (Delta_high/gamma)**2
                denom_high = 1 + (Delta_high/gamma)**2 + (Omega**2/gamma/Gamma_r)
                sigz_vals[idx] = -1*(fit_params['P_low']*num_low/denom_low + (1-fit_params['P_low'])*num_high/denom_high)
            if freq_choice is not None:
                Delta_low = 2*np.pi*(freq_choice - fit_params['f0'] +0.5*fit_params['chi'])
                Delta_high = 2*np.pi*(freq_choice - fit_params['f0'] -0.5*fit_params['chi'])
                num_low = 1 + (Delta_low/gamma)**2
                denom_low = 1 + (Delta_low/gamma)**2 + (Omega**2/gamma/Gamma_r)
                num_high = 1 + (Delta_high/gamma)**2
                denom_high = 1 + (Delta_high/gamma)**2 + (Omega**2/gamma/Gamma_r)
                sigz_val_choice = -1*(fit_params['P_low']*num_low/denom_low + (1-fit_params['P_low'])*num_high/denom_high)
        else:
            for idx, f in enumerate(self.data['freqs']):
                Delta = 2*np.pi*(f - fit_params['f0'])
                num = 1 + (Delta/gamma)**2
                denom = 1 + (Delta/gamma)**2 + (Omega**2/gamma/Gamma_r)
                sigz_vals[idx] = -1*num/denom
            if freq_choice is not None:
                Delta = 2*np.pi*(freq_choice - fit_params['f0'])
                num = 1 + (Delta/gamma)**2
                denom = 1 + (Delta/gamma)**2 + (Omega**2/gamma/Gamma_r)
                sigz_val_choice = -1*num/denom
            
        self.analysis_results['sigz'] = sigz_vals
        self.analysis_results['f0'] = fit_params['f0']
        self.analysis_results['P1'] = (1 + self.analysis_results['sigz']) / 2
        if plot:
            self.plot_sigz(use_NonGl0_result=use_NonGl0_result, freq_choice=freq_choice, sigz_val_choice=sigz_val_choice)
        return sigz_vals, sigz_val_choice
    

    def plot_sigz(self, use_NonGl0_result=True, freq_choice=None, sigz_val_choice=None):
        ## Renormalize sigma z to get excitation probability
        if 'sigz' not in self.analysis_results:
            self.calc_sigz(use_NonGl0_result=use_NonGl0_result)
        if 'sigz' not in self.analysis_results:
            print("Error: No sigz values to plot.")
            return None
        ex_prob = self.analysis_results['P1']
        plt.figure()
        plt.plot(self.data['freqs'], ex_prob, label=r'P(1)', 
                 color='purple', lw=2)
        plt.axvline(self.analysis_results['f0'], color='gray', ls='--')
        max_idx = np.argmax(ex_prob)
        plt.scatter(self.data['freqs'][max_idx], ex_prob[max_idx], 
                    color='red', zorder=5)
        plt.text(self.data['freqs'][max_idx], ex_prob[max_idx],
                 f'  Max Excitation\n  {self.data["freqs"][max_idx]/1e9:.3f} GHz, {ex_prob[max_idx]:.3f}',
                 verticalalignment='top', color='red')
        if freq_choice is not None and sigz_val_choice is not None:
            ex_prob_choice = (1 + sigz_val_choice) / 2
            plt.axvline(x=freq_choice, zorder=5)
            plt.text(freq_choice, ex_prob_choice,
                     f'  Chosen Freq\n  {freq_choice/1e9:.3f} GHz, {ex_prob_choice:.3f}',
                     verticalalignment='bottom', color='blue')
        ## Split y axis, plot res
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('P(1)')
        plt.title('Qubit Population vs Frequency')
        plt.legend()
        plt.show()
        return