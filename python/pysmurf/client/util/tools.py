#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf util tools
#-----------------------------------------------------------------------------
# File       : pysmurf/util/tools.py
# Created    : 2018-08-29
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import numpy as np
from scipy.optimize import curve_fit

def skewed_lorentzian(x, bkg, bkg_slp, skw, mintrans, res_f, Q):
    """ Skewed Lorentzian model.

    Parameters
    ----------
    x : float
        The x-data to build the skewed Lorentzian
    bkg : float
        The DC value of the skewed Lorentzian
    bkg_slp : float
        The slope of the skewed Lorentzian
    skw : float
        The skewness of the Lorentzian
    mintrans : float
        The minimum of the trans. This is associated with the skewness term.
    res_f : float
        The center frequency of the resonator (the center of the Lorentzian)
    Q : float
        The Q of the resonator

    Returns
    -------
    float
        The model of the Lorentzian
    """
    return bkg + bkg_slp*(x-res_f)-(mintrans+skw*(x-res_f))/\
        (1+4*Q**2*((x-res_f)/res_f)**2)

def fit_skewed_lorentzian(f, mag):
    """ Fits frequency and magnitude data with a skewed lorentzian.

    Args
    ----
    f : float array
        The frequency array
    mag : float array
        The resonator response array

    Returns
    -------
    fit_params : float array
        The fit parameters
    """

    # define the initial values
    bkg = (mag[0]+mag[-1])/2
    bkg_slp = (mag[-1]-mag[0])/(f[-1]-f[0])
    skw = 0
    mintrans = bkg-mag.min()
    res_f = f[mag.argmin()]
    Q = 1e4

    low_bounds = [bkg/2, -1e-3, -1, 0, f[0], 1e2]
    up_bounds = [bkg*2, 1e-3, 1, 30, f[-1], 1e5]

    try:
        popt, pcov = curve_fit(skewed_lorentzian, f, mag,
            p0=[bkg, bkg_slp, skw,mintrans, res_f, Q], method='lm')
        if popt[5] < 0:
            popt, pcov = curve_fit(skewed_lorentzian, f, mag,
                p0=[bkg, bkg_slp, skw, mintrans, res_f, Q],
                bounds=(low_bounds, up_bounds))
    except RuntimeError:
        popt = np.zeros((6,))
    except ValueError:
        popt = np.zeros((6,))

    return popt


def limit_phase_deg(phase, minphase=-180):
    """ Limits the phase in degrees

    Brazenly stolen from
    https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees

    Args
    ----
    phase : float
        The input phase
    minphase : float
        The minimum phase

    Returns
    -------
    phase_limited : float
        The phase information with the limited phase
    """
    phase_limited = np.copy(phase)
    while phase_limited <= minphase:
        phase_limited += 360
    while phase_limited > minphase + 360:
        phase_limited -= 360
    return phase_limited

def P_singleMode(f_center, bw, T):
    '''
    Optical power in a single mode in a bandwidth bw centered on frequency
    f_center from an optical load of temperature T.  SI units.

    Args
    ----
    f_center : float
        The center frequency
    bw : float
        The bandwidth in SI units
    T : float
        The temperature

    Returns
    -------
    float
        The optical power
    '''
    h = 6.63e-34
    kB = 1.38e-23
    df = bw/1000.
    f_array = np.arange(f_center-bw/2., f_center+bw/2.+df, df)
    P = 0.

    # Integrate over frequency bandwidth
    for i in range(len(f_array)):
        f = f_array[i]
        P += df*h*f/(np.exp(h*f/(kB*T))-1.)
    return P

def dPdT_singleMode(f_center, bw, T):
    '''
    Change in optical power per change in temperature (dP/dT) in a single mode
    in a bandwidth bw centered on frequency f_center from an optical load of
    temperature T. SI units.
    '''
    dT = T/1e6
    dP = P_singleMode(f_center, bw, T+dT) - P_singleMode(f_center, bw, T)
    return dP/dT

def load_yaml(filename):
    """ Load the yml yaml file

    Args
    ----
    filename : str
        Full path to the yaml file

    Returns
    -------
    yaml_file_object
        The yaml file
    """
    import yaml

    with open(filename, 'r') as stream:
        dat = yaml.safe_load(stream)

    return dat

def yaml_parse(yml, cmd):
    """ Gets the values out of the yaml file

    Args
    ----
    yml : yaml_file
        The input yaml file, loaded with load_yaml
    cmd : str
        The full epics path in the yaml file

    Returns
    -------
    val
        The value associated with the requested cmd
    """
    cmd = cmd.split(':')[1:]  # First is epics root. Throw out

    def get_val(yml, c):
        """ Extracts the values.

        This is a convenience function that calls itself recursively.

        Args
        ----
        yml : yaml_file
            The input yaml_file
        c : str
            The epics path

        Returns
        -------
        val
            The value associated with input param c
        """
        if np.size(c) == 1 and c[0] in yml.keys():
            return yml[c[0]]
        elif np.size(c) > 1 and c[0] in yml.keys():
            return get_val(yml[c[0]], c[1:])
        return np.nan

    return get_val(yml, cmd)


def utf8_to_str(d):
    """
    Many of the rogue variables are returned as UTF8 formatted byte
    arrays by default. This function changes them from UTF8 to a
    string

    Args
    ----
    d : int array
        An integer array with each element equal to a character.

    Returns
    -------
    str
        The string associated with input d.
    """
    return ''.join([str(s, encoding='UTF-8') for s in d])
