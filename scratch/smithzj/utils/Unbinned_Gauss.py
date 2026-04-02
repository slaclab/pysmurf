import numpy as np
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pylab as plt
# pip install zfit
import zfit 

# Semi-poissonian uncertainty
def counting_err(n):
    err_0 = np.sqrt(n)
    #err_0[n<10] = np.sqrt(10)
    err_0[n==0] = 1
    return err_0

# Iterative zfitting procedure for gaussian
# lower guess 
def zfit_gauss(data, low_guess, upp_guess, height, binnum, plotsome=False):
    mu_gauss.lower = low_guess-5
    mu_gauss.upper = upp_guess+5
    mu_gauss.set_value(np.mean(data[(data>low_guess)*(data<upp_guess)]))
    
    sigma_gauss.set_value(np.std(data[(data>low_guess)*(data<upp_guess)]))
    
    for hh in range(10):
        obs = zfit.Space('x', limits=(low_guess,upp_guess))
        zdata = zfit.Data.from_numpy(obs=obs, array=data[(data>low_guess)*(data<upp_guess)])
        num = np.count_nonzero((data>low_guess)*(data<upp_guess))
        region_1 = np.linspace(low_guess,upp_guess,5000,dtype=np.float64)
        #region_2 = np.linspace(low_guess-5,upp_guess+5,5000,dtype=np.float64)
        
        if num <= 20:
            print('too few events')
            return None, region_1, obs, num, None, None, None
        
        # model building, pdf creation
        gauss = zfit.pdf.Gauss(mu=mu_gauss, sigma=sigma_gauss, obs=obs)
        bins = np.linspace(low_guess,upp_guess,binnum)

        # create NLL
        nll = zfit.loss.UnbinnedNLL(model=gauss, data=zdata)
    
        # # create a minimizer
        minimizer = zfit.minimize.Minuit()
        result_gauss = minimizer.minimize(nll)

        # or here with minos
        param_errors_asymetric, new_result_gauss = result_gauss.errors()
        
        vals, bins = np.histogram(data[(data>low_guess)*(data<upp_guess)],bins=bins)
        bincent = 0.5*(bins[:-1]+bins[1:])
        y_fit = zfit.run(gauss.pdf(bincent, norm_range=obs))
        chi2 = sum(((vals - (y_fit*num*(bins[1]-bins[0])))**2)/(counting_err(vals)**2))
        
        new_low_guess = result_gauss.params[mu_gauss]['value']-result_gauss.params[sigma_gauss]['value']*np.sqrt(-2*np.log(height))
        new_upp_guess = result_gauss.params[mu_gauss]['value']+result_gauss.params[sigma_gauss]['value']*np.sqrt(-2*np.log(height))
        
        if plotsome and hh==9:
            y_fit = zfit.run(gauss.pdf(region_1, norm_range=obs))
            bins = np.linspace(low_guess,upp_guess,binnum)
            plt.hist(data[(data>low_guess)*(data<upp_guess)],bins=bins,histtype='step',linewidth=2)
            plt.plot(region_1,y_fit*num*(bins[1]-bins[0]))
            plt.axvline(x=low_guess,c='b')
            plt.axvline(x=upp_guess,c='b')
            plt.axvline(x=new_low_guess,c='r')
            plt.axvline(x=new_upp_guess,c='r')
            plt.axvline(x=0.5*new_low_guess+0.5*low_guess,linestyle=':')
            plt.axvline(x=0.5*new_upp_guess+0.5*upp_guess,linestyle=':')
            plt.show()
        
        low_guess = 0.5*new_low_guess+0.5*low_guess
        upp_guess = 0.5*new_upp_guess+0.5*upp_guess
    if plotsome:
        print(result_gauss)
    
    if not result_gauss.valid:
        print('result not valid')
        return None, region_1, obs, num, None, None
    
    return gauss, region_1, obs, num, result_gauss, chi2, param_errors_asymetric


def TESTzfit_gauss(data, low_guess, upp_guess, height, binnum,mu_gauss, sigma_gauss,  plotsome=False):
    mu_gauss.lower = low_guess-5
    mu_gauss.upper = upp_guess+5
    mu_gauss.set_value(np.mean(data[(data>low_guess)*(data<upp_guess)]))
    
    sigma_gauss.set_value(np.std(data[(data>low_guess)*(data<upp_guess)]))
    
    for hh in range(10):
        obs = zfit.Space('x', limits=(low_guess,upp_guess))
        zdata = zfit.Data.from_numpy(obs=obs, array=data[(data>low_guess)*(data<upp_guess)])
        num = np.count_nonzero((data>low_guess)*(data<upp_guess))
        region_1 = np.linspace(low_guess,upp_guess,5000,dtype=np.float64)
        #region_2 = np.linspace(low_guess-5,upp_guess+5,5000,dtype=np.float64)
        
        if num <= 20:
            print('too few events')
            return None, region_1, obs, num, None, None, None
        
        # model building, pdf creation
        gauss = zfit.pdf.Gauss(mu=mu_gauss, sigma=sigma_gauss, obs=obs)
        bins = np.linspace(low_guess,upp_guess,binnum)
    
        # create NLL
        nll = zfit.loss.UnbinnedNLL(model=gauss, data=zdata)
        # # create a minimizer
        minimizer = zfit.minimize.Minuit()
        result_gauss = minimizer.minimize(nll)
    
        # or here with minos
        param_errors_asymetric, new_result_gauss = result_gauss.errors()
        
        vals, bins = np.histogram(data[(data>low_guess)*(data<upp_guess)],bins=bins)
        bincent = 0.5*(bins[:-1]+bins[1:])
        y_fit = zfit.run(gauss.pdf(bincent, norm_range=obs))
        chi2 = sum(((vals - (y_fit*num*(bins[1]-bins[0])))**2)/(counting_err(vals)**2))
        
        new_low_guess = result_gauss.params[mu_gauss]['value']-result_gauss.params[sigma_gauss]['value']*np.sqrt(-2*np.log(height))
        new_upp_guess = result_gauss.params[mu_gauss]['value']+result_gauss.params[sigma_gauss]['value']*np.sqrt(-2*np.log(height))
        
        if plotsome and hh==9:
            y_fit = zfit.run(gauss.pdf(region_1, norm_range=obs))
            bins = np.linspace(low_guess,upp_guess,binnum)
            plt.hist(data[(data>low_guess)*(data<upp_guess)],bins=bins,histtype='step',linewidth=2)
            plt.plot(region_1,y_fit*num*(bins[1]-bins[0]))
            plt.axvline(x=low_guess,c='b')
            plt.axvline(x=upp_guess,c='b')
            plt.axvline(x=new_low_guess,c='r')
            plt.axvline(x=new_upp_guess,c='r')
            plt.axvline(x=0.5*new_low_guess+0.5*low_guess,linestyle=':')
            plt.axvline(x=0.5*new_upp_guess+0.5*upp_guess,linestyle=':')
            plt.show()
        
        low_guess = 0.5*new_low_guess+0.5*low_guess
        upp_guess = 0.5*new_upp_guess+0.5*upp_guess
    if plotsome:
        print(result_gauss)
    
    if not result_gauss.valid:
        print('result not valid')
        return None, region_1, obs, num, None, None
    
    return gauss, region_1, obs, num, result_gauss, chi2, param_errors_asymetric, nll