import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt
import h5py
from functools import partial
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os
print(os.getcwd())
from Classes.ResonanceFitResult import *
#from DaqUtils.UsrpFunctions import read_vna

def removecable(f, z, tau, f1):
    """
    returns:
        z_no_cable:  z with the cable delay factor removed (guessing tau, relative to f1?)
    """
    z_no_cable = np.array(z)*np.exp(2j*np.pi*(np.array(f)-f1)*tau)
    return z_no_cable

def estpara(f, z, fr_0):
    """
    returns:
        f0_est:  The estimated center frequency for this resonance
        Qr_est:  The estimated total quality factor
        id_f0:   The estimated center frequency in index number space
        id_BW:   The 3dB bandwidth in index number space
    """

    edge_data_f = np.hstack((f[:int(len(f)/10)],f[-int(len(f)/10):]))
    edge_data_z = np.hstack((z[:int(len(f)/10)],z[-int(len(f)/10):]))

    realfit = np.polyfit(edge_data_f,edge_data_z.real,1)
    imagfit = np.polyfit(edge_data_f,edge_data_z.imag,1)
    zfinder = np.sqrt((z.real-(realfit[1]+f*realfit[0]))**2+(z.imag-(imagfit[1]+f*imagfit[0]))**2)
    edge_val = np.mean(zfinder)
    zfinder = (zfinder+np.append(edge_val,zfinder[:-1])+np.append(zfinder[1:],edge_val)+np.append([edge_val,edge_val],zfinder[:-2])+np.append(zfinder[2:],[edge_val,edge_val]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1)+np.append([1,1],zfinder[:-2])+np.append(zfinder[2:],[1,1]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1))/3
    left_trim = np.argmin(zfinder[f<fr_0])
    right_trim = np.argmin(abs(f-fr_0)) + np.argmin(zfinder[f>=fr_0])
    zfinder = zfinder - min(zfinder[left_trim], zfinder[right_trim])
    id_f0 = left_trim + np.argmax(zfinder[left_trim:right_trim+1])

    id_BW_left = left_trim + np.argmin(abs(abs(z[left_trim:id_f0]-z[id_f0])-abs(z[left_trim:id_f0]-np.mean(edge_data_z))))
    id_BW_right = id_f0 + np.argmin(abs(abs(z[id_f0:right_trim+1]-z[id_f0])-abs(z[id_f0:right_trim+1]-np.mean(edge_data_z))))

    if False:
        plt.figure(1)
        #plt.plot(f_f0_finder,abs(dz_f0_finder))
        plt.plot(f[left_trim:right_trim+1], abs(z[left_trim:right_trim+1]),'.')
        plt.plot(f, zfinder, '.')
        plt.plot(f[left_trim:right_trim+1], zfinder[left_trim:right_trim+1], '.')
        plt.axvline(x=fr_0)
        plt.axvline(x=f[id_f0], c="g")
        #plt.axvline(x=f[id_3db_left], color="red")
        #plt.axvline(x=f[id_3db_right], color="red")
        plt.axvline(x=f[id_BW_left], c="r")
        plt.axvline(x=f[id_BW_right], c="r")
        plt.axhline(y=0)
        #plt.axhline(y=z_3db, color="red")

        plt.figure(2)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag)
        plt.plot(realfit[1]+f*realfit[0],imagfit[1]+f*imagfit[0])
        plt.plot(z[id_f0].real,z[id_f0].imag,'o',c='g')
        plt.plot(z[id_BW_left].real,z[id_BW_left].imag,'o',c='r')
        plt.plot(z[id_BW_right].real,z[id_BW_right].imag,'o',c='r')
        plt.show()

    f0_est = f[id_f0]
    #id_BW = 2*np.mean([abs(id_f0-id_3db_left), abs(id_f0-id_3db_right)])
    id_BW = id_BW_right-id_BW_left
    #Qr_est = f0_est/(2*np.mean([abs(f[id_f0]-f[id_3db_left]), abs(f[id_f0]-f[id_3db_right])]))
    Qr_est = f0_est/(f[id_BW_right]-f[id_BW_left])

    return f0_est, Qr_est, id_f0, id_BW

def circle2(z):
    # == METHOD 2b ==
    # "leastsq with jacobian"
    x = z.real
    y = z.imag
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt(((x-xc)**2)+((y-yc)**2))

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri-Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_estimate = x_m, y_m
    center_2b, ier = opt.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b              # circle center
    Ri_2b = calc_R(*center_2b)            # distance of each data point from center_2b
    R_2b = Ri_2b.mean()                   # average Ri_2b, used as predicted radius
    residu_2b = sum((Ri_2b - R_2b)**2)    # residual?

    zc =  center_2b[0]+center_2b[1]*1j

    t = np.arange(0,2*np.pi,0.002)
    xcirc = center_2b[0]+R_2b*np.cos(t)
    ycirc = center_2b[1]+R_2b*np.sin(t)

    return residu_2b, zc, R_2b

def fitphase2(f,z,zc,fr,Qr,z_off):
    z_E3 = z*np.exp(-1j*np.angle(z_off))
    zc_E3 = zc*np.exp(-1j*np.angle(z_off))
    z_off_E3 = z_off*np.exp(-1j*np.angle(z_off))
    phi = np.pi + np.angle(zc_E3-z_off_E3)
    phi = np.angle(np.exp(1j*phi))
    z_no_phi = (z_E3-zc_E3)*np.exp(-1j*phi)

    def no_phi_eq(f_F,fr_F,Qr_F):
        return 4*Qr*(1-f_F/fr_F)/(1-4*Qr_F*Qr_F*((1-f_F/fr_F)**2))

    presults, pcov = opt.curve_fit(no_phi_eq,f,z_no_phi.imag/z_no_phi.real,p0=[fr,Qr])
    #print presults

    if False:
        plt.figure(1)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z_E3.real,z_E3.imag)
        plt.plot(zc_E3.real,zc_E3.imag,'*')
        plt.plot(z_off_E3.real,z_off_E3.imag,'o',fillstyle='none',markersize=15)
        plt.plot(z_no_phi.real,z_no_phi.imag)

        plt.figure(2)
        plt.plot(f,z_no_phi.imag/z_no_phi.real)
        #plt.plot(f,no_phi_eq(f,fr,Qr))
        plt.plot(f,no_phi_eq(f,presults[0],presults[1]))

        plt.show()

    return presults[1], presults[0], phi

def trimdata(f, z, f0, Q, numspan=2):
    f0id = np.argmin(abs(f-f0))
    ind_BW =  (f0/Q)/((f[-1]-f[0])/(len(f)-1))
    idstart = int(math.floor(max(f0id-(ind_BW*numspan), 0)))
    idstop  = int(math.floor(min(f0id+(ind_BW*numspan), len(f))))
    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return fb, zb

def phasefunc(f, Qr, fr, phi, amplitude, sign=1):
    # Jiansong's equation 4.42 with theta = arg(z) + phi from figure E.3
    argz = -phi-amplitude*np.arctan(sign*2*Qr*((f-fr)/fr))
    return argz

def fitphase(f, z, f0g, Qg):
    argz = np.angle(z, deg=False)
    fstep = (f[-1]-f[0])/(len(f)-1)

    fitphase_plotting = False
    if fitphase_plotting:
        plt.figure(3)
        plt.axhline(y=2*np.pi, color="black")
        plt.axhline(y=1*np.pi, color="0.25")
        plt.axhline(y=0, color="0.75")
        plt.axhline(y=-1*np.pi, color="0.25")
        plt.axhline(y=-2*np.pi, color="black")
        plt.plot(f,argz,'.-', label='raw')

    for i in range(len(argz)):
        if i > 1 and abs(argz[i]-argz[i-1]) > (0.5*np.pi):
            fresult = opt.curve_fit(partial(phasefunc, sign=1),f[:i],argz[:i],p0=[Qg, f0g, np.pi,2],bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, max(2, ((max(argz)-min(argz))/np.pi))]))
            guide = phasefunc(f,fresult[0][0],fresult[0][1],fresult[0][2],fresult[0][3])
            argz[i:] += -2*np.pi*round((argz[i]-guide[i+1])/(2*np.pi))

    if fitphase_plotting:
        plt.plot(f, argz, '.-', label='adjusted')

    while np.mean(argz)>np.pi:
        argz = argz - 2*np.pi
    while np.mean(argz)<-np.pi:
        argz = argz + 2*np.pi

    if fitphase_plotting:
        plt.plot(f, argz, '.-', label='adjusted+shifted')

    phi0 = -np.mean(argz)
    d_height = max(argz)-min(argz)
    tan_max_height = max(2, (d_height/np.pi))

    # using robust(?) fit from curve fit
    if np.mean(argz[int(len(argz)/2):]) <= np.mean(argz[:int(len(argz)/2)]):
        fresult = opt.curve_fit(partial(phasefunc, sign=1), f, argz, p0=[Qg, f0g, phi0, 2], bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, tan_max_height]))
    else:
        fresult = opt.curve_fit(partial(phasefunc, sign=-1), f, argz, p0=[Qg, f0g, phi0, 2], bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, tan_max_height]))

    if fitphase_plotting:
        plt.plot(f, phasefunc(f,fresult[0][0],fresult[0][1],fresult[0][2],fresult[0][3],sign=1), '-', label='fit')
        plt.legend()
        plt.show()

    return fresult[0][0],fresult[0][1],fresult[0][2]

def resfunc3(f, fr, Qr, Qc_hat_mag, a, phi, tau):
    """A semi-obvious form of Gao's S21 function. e^(2j*pi*fr*tau) is incorporated into a."""
    S21 = a*np.exp(-2j*np.pi*(f-fr)*tau)*(1-(((Qr/Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return S21

def resfunc8(f_proj, fr, Qr,  Qc_hat_mag, a_real, a_imag, phi, tau, Imtau):
    """An alternate resfunc3 with all real inputs. Re(S21) is projected onto negative frequency space."""
    f = abs(f_proj)
    S21 = (a_real+1j*a_imag)*np.exp(-2j*np.pi*(f-fr)*(tau+1j*Imtau))*(1-(((Qr/ Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    S21 = np.array(S21)
    real_S21 = S21.real
    real_S21[f_proj>0] = 0
    imag_S21 = S21.imag
    imag_S21[f_proj<0] = 0
    return real_S21 + imag_S21

def roughfit(f, z, fr_0, fit_res_obj=None, plot=False):

    edge_data_f = np.hstack((f[:int(len(f)/100)],f[-int(len(f)/100):]))
    edge_data_z = np.hstack((z[:int(len(f)/100)],z[-int(len(f)/100):]))

    id_f0 = np.argmin(abs(f-fr_0))

    tau_fit = np.polyfit(edge_data_f-f[id_f0],np.angle(edge_data_z/z[id_f0]),1)
    tau_1 = tau_fit[0]/(-2*np.pi)

    if plot:
        plt.plot(edge_data_f-f[id_f0],np.angle(edge_data_z/z[id_f0]),'.')
        plt.plot(edge_data_f-f[id_f0],tau_fit[0]*(edge_data_f-f[id_f0])+tau_fit[1])
        plt.show()

    tau_fit = np.polyfit(edge_data_f-f[id_f0],np.log(abs(edge_data_z/z[id_f0])),1)
    Imtau_1 = tau_fit[0]/(2*np.pi)

    # remove cable term
    z1 = removecable(f, z, tau_1+1j*Imtau_1, fr_0)

    if plot:
        #print tau_1
        plt.figure(1)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag)
        plt.plot(z1.real,z1.imag)
        plt.figure(2)
        plt.plot(f,abs(z))
        plt.plot(f,abs(z1))
        #plt.show()

    # estimate f0 (pretty good), Q (very rough)
    f0_est, Qr_est, id_f0, id_BW = estpara(f,z1,fr_0)

    ## Save the estimates to our class instance
    if (fit_res_obj is not None):
        fit_res_obj.f0_est = f0_est
        fit_res_obj.Qr_est = Qr_est
        fit_res_obj.id_f0  = id_f0
        fit_res_obj.id_BW  = id_BW

    # fit circle using trimmed data points
    id1 = max(id_f0-int(0.5*id_BW), 0)
    id2 = min(id_f0+int(0.5*id_BW), len(f))
    if len(range(id1, id2)) < 3:
        id1 = 0
        id2 = len(f)

    residue, zc, r = circle2(z1[id1:id2])

    # rotation and traslation to center
    z1b = z1*np.exp(-1j*np.angle(zc, deg=False))
    z2 = (zc-z1)*np.exp(-1j*np.angle(zc, deg=False))

    if plot:
        plt.figure(2)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag,'o', color=None, label='z')
        plt.plot(z1.real,z1.imag,'o', color=None, label='z1')
        #plt.plot(z1b.real,z1b.imag,'o', color=None, label='z1b')
        #plt.plot(z2.real,z2.imag,'o', color=None, label='z2')
        #plt.plot(z2[np.argmin(abs(f-f0_est))].real,z2[np.argmin(abs(f-f0_est))].imag,'o', color='orange', label='f0 est')
        plt.legend()
        plt.show()

    # fit phase
    #Q, f0, phi = fitphase(f, z2, f0_est, Qr_est)

    z_off = np.mean(np.hstack((z1[:int(len(f)/10)],z1[-int(len(f)/10):])))
    Q, f0, phi = fitphase2(f,z1,zc,f0_est,Qr_est,z_off)

    Qc = abs(z_off)*Q/(2*r)

    if plot:
        z_rough = z_off*np.exp(-2j*np.pi*(f-f0)*(tau_1+1j*Imtau_1))*(1-((Q/Qc)*np.exp(1j*phi))/(1+2j*Q*((f-f0)/f0)))
        plt.figure(1)
        plt.plot(z.real,z.imag)
        plt.plot(z_rough.real,z_rough.imag)
        plt.plot(z_off.real,z_off.imag,'*')
        plt.figure(2)
        plt.plot(f,z.real)
        plt.plot(f,z.imag)
        plt.plot(f,z_rough.real)
        plt.plot(f,z_rough.imag)
        plt.show()

    ## Create a dictionary for the output
    result = {  "f0"    : f0, 
                "Q"     : Q,
                "phi"   : phi,
                "zOff"  : z_off,
                "Qc"    : Qc,
                "tau1"  : tau_1,
                "Imtau1": Imtau_1}

    ## Write the rough fit result to the fit result class instance
    if (fit_res_obj is not None):
        fit_res_obj.rough_result["f0"]     = result["f0"]
        fit_res_obj.rough_result["Q"]      = result["Q"]
        fit_res_obj.rough_result["phi"]    = result["phi"]
        fit_res_obj.rough_result["zOff"]   = result["zOff"]
        fit_res_obj.rough_result["Qc"]     = result["Qc"]
        fit_res_obj.rough_result["tau1"]   = result["tau1"]
        fit_res_obj.rough_result["Imtau1"] = result["Imtau1"]

    return result

## Note that fine fit fits for Qr and Qc, from which Qi is calculated
def finefit(f, z, fr_0, restrict_fit_MHz=None, fit_res_obj=None, show_plots=False, verbose=True):
    """
    finefit fits f and z to the resonator model described in Jiansong's thesis

    Input parameters:
                  f: frequencys [GHz]
                  z: complex impedence
               fr_0: initially predicted fr [GHz]

    Returns:
        fr_fine, Qr_fine, Qc_hat_mag_fine, a_fine, phi_fine, tau_fine, Qc_fine
    """

    ## First trim the data if specified
    # if restrict_fit_MHz is not None:
    #     fmin = fr_0 - (restrict_fit_MHz*1e-3/2.)
    #     fmax = fr_0 + (restrict_fit_MHz*1e-3/2.)
    #     ft = f[ (f>fmin)&(f<fmax) ]
    #     zt = z[ (f>fmin)&(f<fmax) ]
    #     f  = ft
    #     z  = zt

    # find starting parameters using a rough fit
    # fr_1, Qr_1, phi_1, a_1, Qc_hat_mag_1, tau_1, Imtau_1 = roughfit(f, z, fr_0)
    rough_res = roughfit(f, z, fr_0, fit_res_obj,  plot=False)

    ## Unpack the result
    fr_1    = rough_res["f0"]
    Qr_1    = rough_res["Q"]
    phi_1   = rough_res["phi"]
    a_1     = rough_res["zOff"]
    Qc_hat_mag_1 = rough_res["Qc"]
    tau_1   = rough_res["tau1"]
    Imtau_1 = rough_res["Imtau1"]

    ## Create array of initial guesses for params
    pGuess = [fr_1, Qr_1, Qc_hat_mag_1, a_1.real, a_1.imag, phi_1, tau_1, Imtau_1]
    if (fit_res_obj is not None):
        fit_res_obj.fine_pguess = pGuess

    # trim data?
    #if False:
    #    fnew, znew = trimdata(f, z, fr_1, Qr_1)
    #    if len(fnew)>10:
    #        f = fnew
    #        z = znew

    # combine x and y data so the fit can go over both simultaneously
    xdata = np.concatenate([-f, f])
    ydata = np.concatenate([z.real, z.imag])
    

    if show_plots:
        plt.figure("v"+str(1))
        plt.plot(abs(xdata[:len(z)]),20*np.log10(abs(ydata[:len(z)]+1j*ydata[len(z):])),'.')
        plt.plot(abs(xdata[:len(z)]),20*np.log10(abs(resfunc8(xdata[:len(z)],fr_1, Qr_1, Qc_hat_mag_1, a_1.real, a_1.imag, phi_1, tau_1, Imtau_1)+1j*resfunc8(xdata[len(z):],fr_1, Qr_1, Qc_hat_mag_1, a_1.real, a_1.imag, phi_1, tau_1, Imtau_1))),label='roughfit')
        # plt.savefig("fig1")
        plt.figure("v"+str(2))
        plt.plot(abs(xdata),ydata,'.')
        plt.plot(abs(xdata),resfunc8(xdata,fr_1, Qr_1, Qc_hat_mag_1, a_1.real, a_1.imag, phi_1, tau_1, Imtau_1),label='roughfit')
        # plt.savefig("fig2")
        plt.figure("v"+str(3))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.plot(z.real,z.imag,'.')
        plt.plot(resfunc3(f,fr_1,Qr_1,Qc_hat_mag_1,a_1,phi_1,tau_1).real,resfunc3(f,fr_1,Qr_1,Qc_hat_mag_1,a_1,phi_1,tau_1).imag,label='roughfit')
        # plt.savefig("fig3")

    fopt, fcov = opt.curve_fit(resfunc8, xdata, ydata, p0=pGuess, bounds=([min(f), 0, 0, -np.inf, -np.inf,-1*np.pi, -np.inf, -np.inf], [max(f), np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, np.inf]))
    ferr = np.sqrt(np.diag(fcov))

    min_S21logmag = np.min(  20*np.log10(abs(resfunc8(xdata[:len(z)],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7])+1j*resfunc8(xdata[len(z):],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7]))) )
    min_frequency = np.abs(xdata[:len(z)][np.argmin(  20*np.log10(abs(resfunc8(xdata[:len(z)],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7])+1j*resfunc8(xdata[len(z):],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7]))) )])

    if show_plots:
        plt.figure("v"+str(1))
        plt.plot(abs(xdata[:len(z)]), 20*np.log10(abs(resfunc8(xdata[:len(z)],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7])+1j*resfunc8(xdata[len(z):],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7]))),label='finefit')
        plt.plot(fopt[0], 20*np.log10(abs(resfunc8(-1.*fopt[0],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7])+1j*resfunc8(fopt[0],fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7]))),'o',label='finefit fr')
        plt.plot(min_frequency, min_S21logmag,'o',label='finefit min', color='purple')
        plt.legend()
        plt.savefig("VNA-S21logmag-fit.png")
        plt.figure("v"+str(2))
        plt.plot(abs(xdata),resfunc8(xdata,fopt[0],fopt[1],fopt[2],fopt[3],fopt[4],fopt[5],fopt[6],fopt[7]),label='finefit')
        plt.legend()
        plt.savefig("VNA-normphasemag-fit.png")
        plt.figure("v"+str(3))
        plt.plot(resfunc3(f,fopt[0],fopt[1],fopt[2],fopt[3]+1j*fopt[4],fopt[5],fopt[6]+1j*fopt[7]).real,resfunc3(f,fopt[0],fopt[1],fopt[2],fopt[3]+1j*fopt[4],fopt[5],fopt[6]+1j*fopt[7]).imag,label='finefit')
        plt.plot(resfunc3(fopt[0],fopt[0],fopt[1],fopt[2],fopt[3]+1j*fopt[4],fopt[5],fopt[6]+1j*fopt[7]).real,resfunc3(fopt[0],fopt[0],fopt[1],fopt[2],fopt[3]+1j*fopt[4],fopt[5],fopt[6]+1j*fopt[7]).imag,'o',label='finefit fr')
        plt.legend()
        plt.savefig("VNA-S21complex-fit.png")
        # plt.show()

    if verbose:
        print("Fr from fit  [GHz]:", fopt[0])
        print("Fr min curve [GHz]:", min_frequency)

    ## Create a dictionary of the result params
    fine_pars = { "f0"    : fopt[0], 
                  "Qr"    : fopt[1],
                  "phi"   : fopt[5],
                  "zOff"  : fopt[3]+1j*fopt[4],
                  "QcHat" : fopt[2],
                  "tau"   : fopt[6]+1j*fopt[7],
                  "Qc"    : fopt[2]/np.cos(fopt[5])}
    fine_errs = { "f0"    : ferr[0], 
                  "Qr"    : ferr[1],
                  "phi"   : ferr[5],
                  "zOff"  : ferr[3]+1j*ferr[4],
                  "QcHat" : ferr[2],
                  "tau"   : ferr[6]+1j*ferr[7],
                  "Qc"    : ferr[2]/np.cos(ferr[5])}

    if (fit_res_obj is not None):
        fit_res_obj.fine_result = fine_pars
        fit_res_obj.fine_errors = fine_errs

    # fr_fine = fparams[0]
    # Qr_fine = fparams[1]
    # Qc_hat_mag_fine = fparams[2]
    # a_fine = fparams[3]+1j*fparams[4]
    # phi_fine = fparams[5]
    # tau_fine = fparams[6] + 1j*fparams[7]

    # Qc_fine =  Qc_hat_mag_fine/np.cos(phi_fine)

    return fine_pars, fine_errs


def peak_search(f, z, fwindow=5e-4, start_f=None, stop_f=None, nsig=3, max_N_peaks=10):
    '''
    ## Perform a search for resonance peaks using a high-pass filter that will
    ## identify points that are above some number of sigma (RMS about the mean)
    ## ARGUMENTS
    ## -              f: array of frequency values [GHz]
    ## -              z: array of corresponding z values
    ## -        fwindow: half the window cut around each resonator before fitting [GHz]
    ## -           nsig: nsig*sigma is the threshold for resonator identification
    ## -    max_N_peaks: if more than this many peaks are identified, call the only peak the minimum of the transmission spectrum
    ## -        start_f: lower bound of resonance identification region [GHz]
    ## -         stop_f: upper bound of resonance identification region [GHz]
    ## RETURNS:
    ## -       peaklist: array of indeces corresponding to the peaks
    ## -            mfz: filtered transmission spectrum
    '''

    ## Extract Nyquist frequency [s]
    nfreq = 1/(2*(abs(f[-1]-f[0])/(len(f)-1)))

    ## The frequency corresponding to the expected window size [s]
    evfreq = 1/(2*fwindow) 

    ## Butter some bread?
    b, a = sig.butter(2, evfreq/nfreq, btype='highpass')

    ## The magnitude of filtered z, The filtfilt part calls a deprication warning for unknown reasons
    mfz = np.sqrt(sig.filtfilt(b, a, z.real)**2 + sig.filtfilt(b, a, z.imag)**2)  

    ## Do some averaging
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0)+np.append([0,0],mfz[:-2])+np.append(mfz[2:],[0,0]))/5
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0))/3

    ## Record the standard deviation of mfz
    bstd = np.std(mfz)

    ## initialize peaklist
    peaklist  = np.array([], dtype = int)

    ## initialize mx below min
    mx = -np.inf
    peak_pos = 0
    mx_pos = np.nan
    lookformax = False
    delta = nsig*bstd
    gamma = 3*np.mean(mfz[mfz<delta])

    ## Definte the frequency space to search if none is provided
    if (start_f is None) or (start_f<f[0]):
        start_f = f[0]
    if (stop_f  is None) or (stop_f >f[-1]):
        stop_f  = f[-1]

    ## find peaks and add them to peaklist
    for i in range(len(mfz)):
        if (f[i] >= start_f)*(f[i] <= stop_f):
            cp = mfz[i]
            if cp >= mx:
                mx = cp
                mx_pos = i
            if lookformax == True:
                if cp < gamma:
                    peak_pos  = mx_pos
                    peaklist  = np.append(peaklist , peak_pos)
                    lookformax = False
            else:
                # if cp > delta and f[i] > (min(f)+2*fwindow):
                if cp > delta:
                    mx = cp
                    mx_pos = i
                    lookformax = True

    ## Handle too many peaks by picking the minimum of the unfiltered transmission
    if (len(peaklist) > max_N_peaks):
        # peaklist = np.array([ np.argmax(20*np.log10(abs(np.array(z)))) ])
        # peaklist = np.array([ np.argmax(mfz) ])
        peaklist = np.array([ np.argmin(20*np.log10(np.array(z))) ])

    return peaklist, mfz


def sweep_fit(f, z, file_fit_obj, additions=[], fwindow=5e-4, nsig=3, max_N_peaks=10, start_f=None, stop_f=None, verbose=False, show_plots=False):
    """
    sweep_fit fits data to the resonator model described in Jiansong's thesis

    Input parameters:
                  f: array of frequency values [GHz]
                  z: array of corresponding z values
       file_fit_obj:
          additions: list of resonances to be manually included [GHz]
            fwindow: see peak_search docstring
               nsig: see peak_search docstring
        max_N_peaks: see peak_search docstring
            start_f: see peak_search docstring
             stop_f: see peak_search docstring

    Returns:
        fr_list, Qr_list, Qc_list, Qi_list (fitted values for each resonator)
    """

    # Sort the datapoints so frequencies are in order
    z_1 = [zs for _,zs in sorted(zip(f,z))]
    f_1 = [fs for fs,_ in sorted(zip(f,z))]
    z = np.array(z_1)
    f = np.array(f_1)

    ## Search for peaks using a filter
    peaklist, mfz = peak_search(f, z, fwindow=fwindow, start_f=start_f, stop_f=stop_f, nsig=nsig, max_N_peaks=max_N_peaks)

    ## add the manually entered frequencies to peaklist
    for added in additions:
        peaklist = np.append(peaklist, np.argmin(abs(f-added)))
    addlist = peaklist

    peaklist = sorted(peaklist)
    if verbose:
        print('Position of identified', len(peaklist), 'peaks (index):', peaklist)
    file_fit_obj.resize_peak_fits(len(peaklist))

    ## ============================================================= ##
    if show_plots:
        ## Create a plot for the logmag vs frequency
        fig = plt.figure(figsize=(9,7))

        # Define the grid
        gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[1, 1])
        gs.update(wspace=0.33, hspace=0.30) 

        # Define the plot array
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        ## Set plot title
        ax0.set_title('Transmission with Resonance Identification')

        ## Plot the unaltered transmission on top panel
        ax0.plot(f, 20*np.log10(abs(np.array(z))))

        ## Plot the bottom panel
        ax1.plot(f, mfz/np.std(mfz))
        
        ## Make the labels
        ax0.set_ylabel(r"|$S_{21}$| [dB]")
        ax1.set_ylabel(r"|filtered z| [$\sigma$]")
        ax1.set_xlabel("Frequency [GHz]")

        ## Set the x-tick markers to be smaller
        plt.setp(ax0.get_xticklabels(), fontsize=10)
        plt.setp(ax1.get_xticklabels(), fontsize=10)

        ## Draw some lines
        ax1.axhline(y=nsig, color="red")#, label="nsig = "+str(nsig))
        # ax1.axhline(y=gamma/np.std(mfz), color="green")
        ax1.axvline(x=start_f, color="gray")
        ax1.axvline(x=stop_f, color="gray")

        ## Draw a point for each resonance found
        ax1.plot(f[peaklist], mfz[peaklist]/np.std(mfz), 'gs', label=str(len(peaklist)-len(additions))+" resonances identified")

        ## Draw a point for each resonance added manually
        if len(additions)>0:
            ax1.plot(f[addlist], mfz[addlist]/np.std(mfz), 'ys', label=str(len(addlist))+" resonances manually added")
        
        ## Draw the legend
        plt.legend()

        ## Create a plot for the logmag vs frequency
        axp = plt.figure(figsize=(9,7)).gca()
        axp.set_aspect('equal', adjustable='box')
        # axp.axvline(x=0, color='gray')
        # axp.axhline(y=0, color='gray')
        axp.set_xlabel(r"$\Re( S_{21})$")
        axp.set_ylabel(r"$\Im( S_{21})$")

    else:
        fig = None
    ## ============================================================= ##

    # initialize the parameter lists
    fr_list  = np.zeros(len(peaklist))
    Qr_list  = np.zeros(len(peaklist))
    Qc_hat_mag_list = np.zeros(len(peaklist))
    Qc_list  = np.zeros(len(peaklist))
    Qi_list  = np.zeros(len(peaklist))
    a_list   = np.zeros(len(peaklist), dtype='complex')
    phi_list = np.zeros(len(peaklist))
    tau_list = np.zeros(len(peaklist), dtype='complex')

    # define the windows around each peak. and then use finefit to find the parameters
    for i in range(len(peaklist)):

        ## Create an instance of a single peak fit result
        this_r = SinglePeakResult(i)
        this_r.f_ctr    = f[peaklist[i]]
        this_r.mfz_ctr  = mfz[peaklist[i]]
        this_r.pk_added = i < len(additions)

        print('Resonance #{}'.format(str(i)))
        curr_pts = (f >= (f[peaklist[i]]-2*fwindow)) & (f <= (f[peaklist[i]]+2*fwindow))
        f_curr = f[curr_pts]
        z_curr = z[curr_pts]

        try:
            fine_pars, fine_errs = finefit(f_curr, z_curr, f[peaklist[i]], fit_res_obj=this_r, verbose=verbose)
            fr_list[i]  = fine_pars["f0"]
            Qr_list[i]  = fine_pars["Qr"]
            Qc_hat_mag_list[i] = fine_pars["QcHat"]
            a_list[i]   = fine_pars["zOff"]
            phi_list[i] = fine_pars["phi"]
            tau_list[i] = fine_pars["tau"] 
            Qc_list[i]  = fine_pars["Qc"]
            Qi_list[i]  = (Qr_list[i]*Qc_list[i])/(Qc_list[i]-Qr_list[i])
        except Exception as e:
            print('      FIT FAILURE:',e)
            fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i], Qc_list[i] = [f[peaklist[i]],1e4,1e5,1,0,0,1e5]
            Qi_list[i] = 0

        ## Now that the subroutines have populated class attributes, let's look at them
        if verbose:
            this_r.show_par_ests()
            this_r.show_rough_result()
            this_r.show_fine_result()

        file_fit_obj.peak_fits[i] = this_r

        fit_discrete = resfunc3(f_curr, fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
        SSE = sum((z_curr.real-fit_discrete.real)**2+(z_curr.imag-fit_discrete.imag)**2)

        ## ============================================================= ##
        if show_plots:
            f_continuous = np.linspace(f_curr[0],f_curr[-1],int(5e3))
            fit      = resfunc3(f_continuous, fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            zrfit    = resfunc3(fr_list[i], fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            # fit_down = resfunc3(f_continuous, fr_list[i], 0.95*Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            # fit_up   = resfunc3(f_continuous, fr_list[i], 1.05*Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])

            ## Draw the fit on the frequency space plot
            # ax0.plot(f_curr, 20*np.log10(abs(z_curr)),'.', label='Data')
            ax0.plot(f_continuous, 20*np.log10(abs(fit)), label='Fit', color='red')
            # ax0.plot(f_continuous, 20*np.log10(abs(fit_down)), label='Fit 0.95Q')
            # ax0.plot(f_continuous, 20*np.log10(abs(fit_up)), label='Fit 1.05Q')
            ax0.plot(fr_list[i], 20*np.log10(abs(zrfit)), '*', markersize=10, color='red', label='$f_{r}$')

            ## Draw the polar plot
            # axp.plot(z_curr.real, z_curr.imag,'.', label='Data')
            axp.plot(fit.real, fit.imag,  label='Fit', color='red')
            # axp.plot(fit_down.real, fit_down.imag,  label='Fit 0.95Q')
            # axp.plot(fit_up.real, fit_up.imag, label='Fit 1.05Q')
            axp.plot(zrfit.real, zrfit.imag, '*', markersize=10, color='red',  label='Fr')
            axp.set_xticks([min(z_curr.real),max(z_curr.real)])
            axp.set_yticks([min(z_curr.imag),max(z_curr.imag)])

            fitwords = "$f_{r}$ = " + str(fr_list[i]) + "\n" + "$Q_{r}$ = " + str(Qr_list[i]) + "\n" + "$Q_{c}$ = " + str(Qc_list[i]) + "\n" + "$Q_{i}$ = " + str(Qi_list[i]) + "\n" + "$\phi_{0}$ = " + str(phi_list[i]) + "\n" + "$a$ = " + str(a_list[i]) + "\n" + r"$\tau$ = " + str(tau_list[i]) + "\n"
            xlim = axp.get_xlim()
            ylim = axp.get_ylim()
            axp.text(xlim[0]+1.01*(xlim[1]-xlim[0]), ylim[0]+0.50*(ylim[1]-ylim[0]), fitwords)
            axp.text(xlim[0]+1.01*(xlim[1]-xlim[0]), ylim[0]+0.85*(ylim[1]-ylim[0]), r"$S_{21}(f)=ae^{-2\pi j(f-fr)\tau}\left [ 1-\frac{\frac{Q_{r}}{|\widehat{Q}_{c}|}e^{j\phi_{0}}}{1+2jQ_{r}(\frac{f-f_{r}}{f_{r}})} \right ]$", fontsize=20)
        ## ============================================================= ##

    return fr_list, Qr_list, Qc_list, Qi_list, fig


def vna_fit(VNA_f, VNA_z, pickedres,show_plots=False,save=False,verbose=False):
    '''
    Expects frequency in GHz
    '''
    ## Get the frequency spacing
    df = VNA_f[1] - VNA_f[0]
    frequency_range = 1e-3
    index_range = int(frequency_range / df / 2)

    ## Create some output containers
    frs = np.zeros(len(pickedres))
    Qrs = np.zeros(len(pickedres))
    Qcs = np.zeros(len(pickedres))

    ## Loop over the number of resonators we are looking to fit
    for MKIDnum in range(len(pickedres)):

        MKID_index = np.argmin(abs(VNA_f-pickedres[MKIDnum]))

        MKID_f = VNA_f[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]
        MKID_z = VNA_z[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]

        res_pars, res_errs = finefit(MKID_f, MKID_z, pickedres[MKIDnum], restrict_fit_MHz=None, show_plots=True, verbose=verbose)

        ## Extract the fit result
        frs[MKIDnum] = res_pars["f0"]
        Qrs[MKIDnum] = res_pars["Qr"]
        Qc_hat       = res_pars["QcHat"]
        a            = res_pars["zOff"]
        phi          = res_pars["phi"]
        tau          = res_pars["tau"]
        Qcs[MKIDnum] = res_pars["Qc"]

        ## Calculate the fit
        fit_z = resfunc3(MKID_f, frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau)
        fr_idx = np.argmin(abs(MKID_f-frs[MKIDnum]))

        if show_plots:
            ax_pol = plt.figure('polar').gca()
            ax_pol.plot(MKID_z.real,MKID_z.imag,ls='',marker='.')
            ax_pol.plot(fit_z.real,fit_z.imag,color='lightgrey')
            ax_pol.plot(fit_z[fr_idx].real,fit_z[fr_idx].imag,marker='*')

            ax_pol.set_aspect('equal', adjustable='box')
            ax_pol.set_xlabel('ADC units')
            ax_pol.set_ylabel('ADC units')
            ax_pol.axvline(x=0, color='gray')
            ax_pol.axhline(y=0, color='gray')

            ax_log = plt.figure('logmag').gca()
            ax_log.plot(MKID_f,20*np.log10(abs(MKID_z)))
            ax_log.plot(MKID_f,20*np.log10(abs(fit_z)))

            ax_log.set_xlabel('Frequency [GHz]')
            ax_log.set_ylabel('ADC units')

        if save:
            plt.savefig('VNA_fit_res'+str(MKIDnum)+'.png')
            plt.close()

    return frs, Qrs, Qc_hat, a, phi, tau, Qcs

def vna_file_fit(filename,pickedres,show_plots=False,save=False,verbose=False):
    pickedres = np.array(pickedres)
    VNA_f, VNA_z = read_vna(filename, decimation=1)
    VNA_f = VNA_f*1e-3 #VNA_f in units of GHz after this line
    return vna_fit(VNA_f, VNA_z, pickedres,show_plots=show_plots,save=save,verbose=verbose)

def sweep_fit_from_file(filename, nsig=3, fwindow=5e-4, chan="S21", additions=[], start_f=None, stop_f=None, h5_rewrite=False, pdf_rewrite=False):
    """
    inputs save_scatter data to sweep_fit, can save the results back to the h5 file

    Input parameters:
           filename: full name of the save_scatter h5 file
               nsig: nsig*sigma is the threshold for resonator identification
            fwindow: half the window cut around each resonator before fitting [GHz]
               chan: channel from save_scatter being analyzed
         h5_rewrite: save fit data to the filename.h5?
        pdf_rewrite: save fit data to the filename.pdf?
          additions: list of resonances to be manually included [GHz]
            start_f: lower bound of resonance identification region [GHz]
             stop_f: upper bound of resonance identification region [GHz]

    Returns:
        Nothing
    """

    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])
    fr_list, Qr_list, Qc_list, Qi_list, fig = sweep_fit(f,z,
        nsig=nsig, fwindow=fwindow, start_f=start_f, stop_f=stop_f,
        additions=additions, filename=filename[:-3], pdf_rewrite=pdf_rewrite)

    # save the lists to fname
    if h5_rewrite:
        with h5py.File(fname, "r+") as fyle:
            if "{}/fr_list".format(chan) in fyle:
                fyle.__delitem__("{}/fr_list".format(chan))
            if "{}/Qr_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qr_list".format(chan))
            if "{}/Qc_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qc_list".format(chan))
            if "{}/Qi_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qi_list".format(chan))
            fyle["{}/fr_list".format(chan)] = fr_list
            fyle["{}/Qr_list".format(chan)] = Qr_list
            fyle["{}/Qc_list".format(chan)] = Qc_list
            fyle["{}/Qi_list".format(chan)] = Qi_list