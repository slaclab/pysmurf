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

    return popt