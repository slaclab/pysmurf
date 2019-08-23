import numpy as np
from scipy.optimize import curve_fit

def skewed_lorentzian(x, bkg, bkg_slp, skw, mintrans, res_f, Q):
    """
    Skewed Lorentzian
    """
    return bkg + bkg_slp*(x-res_f)-(mintrans+skw*(x-res_f))/\
        (1+4*Q**2*((x-res_f)/res_f)**2)

def fit_skewed_lorentzian(f, mag):
    """
    Fits frequency and magnitude data with a skewed lorentzian
    """
    
    # define the initial values
    bkg = (mag[0]+mag[-1])/2
    bkg_slp = (mag[-1]-mag[0])/(f[-1]-f[0])
    skw = 0
    mintrans = bkg-mag.min()
    res_f = f[mag.argmin()]
    Q = 1e4

    low_bounds = [bkg/2,-1e-3,-1,0,f[0],1e2]
    up_bounds = [bkg*2,1e-3,1,30,f[-1],1e5]

    try:
        popt,pcov = curve_fit(skewed_lorentzian, f, mag, 
            p0=[bkg,bkg_slp,skw,mintrans,res_f,Q], method='lm')
        if popt[5]<0:
            popt, pcov = curve_fit(skewed_lorentzian, f, mag,
                p0=[bkg,bkg_slp,skw,mintrans,res_f,Q],
                bounds=(low_bounds,up_bounds))
    except RuntimeError:
        popt=np.zeros((6,))
    except ValueError:
        popt=np.zeros((6,))

    return popt


def limit_phase_deg(phase,minphase=-180):
    """
    Brazenly stolen from                                                         
    https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
    """
    newPhase=phase
    while newPhase<=minphase:
        newPhase+=360
    while newPhase>minphase+360:
        newPhase-=360
    return newPhase

def P_singleMode(f_center,bw,T):
    '''
    Optical power in a single mode in a bandwidth bw centered on frequency 
    f_center from an optical load of temperature T.  SI units. 
    '''
    h=6.63e-34
    kB=1.38e-23
    df=bw/1000.
    f_array = np.arange(f_center-bw/2.,f_center+bw/2.+df,df)
    P = 0.
    for i in range(len(f_array)):
        f=f_array[i]
        P += df*h*f/(np.exp(h*f/(kB*T))-1.)
    return P

def dPdT_singleMode(f_center,bw,T):
    '''                                 
    Change in optical power per change in temperature (dP/dT) in a single mode 
    in a bandwidth bw centered on frequency f_center from an optical load of 
    temperature T. SI units.                                
    '''
    dT = T/1e6
    dP = P_singleMode(f_center,bw,T+dT) - P_singleMode(f_center,bw,T)
    return dP/dT

def load_yaml(filename):
    """
    """
    import yaml

    with open(filename, 'r') as stream:
        dat = yaml.safe_load(stream)

    return dat

def yaml_parse(yml, cmd):
    """
    """
    cmd = cmd.split(':')[1:]  # First is epics root. Throw out

    def get_val(yml, c):
        if np.size(c) == 1 and c[0] in yml.keys():
            return yml[c[0]]
        elif np.size(c) > 1 and c[0] in yml.keys():
            return get_val(yml[c[0]], c[1:])
        else:
            return np.nan

    return get_val(yml, cmd)



