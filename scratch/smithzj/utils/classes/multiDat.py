import numpy as np
import os 
import matplotlib
import matplotlib.pylab as plt
import glob
import re
import h5py
import json
import zfit 
from scipy.optimize import curve_fit


from utils.ResonanceFitter import *
from utils.data_collection_utils import rotate2IdealSemiCompact, rotate2IdealCompact, fitMins, fitMaxs, pulse_choices

import utils.MB_equations as MBe

class MultiDataProcessing:
    def __init__(self, debugDatAnalysis_list, fs,num_pulses, pre_trigger_s, metadata=None):
        self.debugList = debugDatAnalysis_list
        self.metadata = metadata  ## TODO, how do you want to load? 
        self.file_specific_templates = {}
        self.pulse_specific_templates = {}
        self.file_specific_PSDs = {}
        self.pulse_specific_PSDs = {}
        self.num_pulses = num_pulses ## TODO set this up
        self.fs = fs ## TODO load this? 
        if num_pulses == 0:
            self.background_pulse = 0
        else: self.background_pulse = num_pulses - 1
        self.pre_trigger_idx =   int(pre_trigger_s  *  self.fs )
        self.row_cuts = {}  
        self.file_cuts = []
        self.makeRowMap()

        self.populateDicts('dff')
        self.populateDicts('dis')
        return
        
    ######## Methods to help with everhtying #######
    
    def runMethodFromList(self, method2run, list_of_arrays,rows_to_exclude=[],  kwargs={}):
        stacked_data = np.concatenate(list_of_arrays, axis=0)
        stacked_data = np.delete(stacked_data, obj=rows_to_exclude, axis=0)
        outputs = method2run(stacked_data, **kwargs)
        return outputs
        
    def subtractMean(self, array2D, start_idx, stop_idx):
        return  array2D - np.mean( array2D[:,start_idx:stop_idx], axis=1, keepdims=True)
                      
    def runMethodOverPulseType(self, method2run, pulse_choice, basis,subtract_mean=False, debug_list_idxs=None, discard_rows=[],kwargs={}):
        assert pulse_choice <= self.num_pulses
        if isinstance(debug_list_idxs, type(None)):
            debug_list_idxs = np.arange(len(self.debugList))
        if pulse_choice == self.background_pulse:
            avg_start_idx = 0
            avg_stop_idx = None
        else:
            avg_start_idx = 0
            avg_stop_idx = int(np.round(self.pre_trigger_idx*.8))
        if subtract_mean: 
            kwargs['subtract_mean'] = True
            kwargs['avg_start_idx'] = avg_start_idx
            kwargs['avg_stop_idx'] = avg_stop_idx
            
        data_list = []
        for idx in debug_list_idxs: 
            datAnalysis = self.debugList[idx]
            data_list.append(datAnalysis.grabPulse(basis, pulse_choice))
        outputs = self.runMethodFromList(method2run, data_list, rows_to_exclude=discard_rows, kwargs=kwargs)
        return outputs

    def getFnameFromIdx(self, idx):
        return self.debugList[idx].file_name
        
    def makeRowMap(self):
        map = {}
        for e, c in enumerate(self.debugList):
            map[e] = {}
            for p in range(self.num_pulses):
                map[e][p] = len(c.grabPulse('dff', p))
        self.map = map

    def mapRow2Data(self, debug_list_idxs, pulse, row):
        row_total = 0 
        for k in debug_list_idxs:
            #print(self.map[k][pulse])
            row_total += self.map[k][pulse]
            #print(row)
            #print(row_total)
            if row_total > row:
                idx_of_interest = k
                row_of_interest = self.map[k][pulse] - row
                break
        return [idx_of_interest, row_of_interest]

    def populateFileSpecificDict(self, method2run, start_dict, basis,subtract_mean=False,kwargs={}):
        for p in range(self.num_pulses): 
            for e in range(len(self.debugList)):
                if e in start_dict.keys():
                    if isinstance(start_dict[e], type({})):
                        #enter this conditional if you this index exists and is a dictionary
                        if basis not in start_dict[e].keys():
                            start_dict[e][basis] = {}
    
                else: start_dict[e] = {basis:{}}
                start_dict[e][basis][p] = self.runMethodOverPulseType(method2run, p, basis, subtract_mean=subtract_mean, debug_list_idxs=[e],kwargs=kwargs)
        
        return start_dict

    def populatePulseSpecificDict(self, method2run, start_dict, basis, subtract_mean=False,  kwargs={}):
        if basis in start_dict.keys():
            if not isinstance(start_dict[basis], type({})):
                start_dict[basis] = {}
        else: start_dict[basis] = {}
            
        for p in range(self.num_pulses):
                
            start_dict[basis][p] = self.runMethodOverPulseType(method2run, p, basis, subtract_mean=subtract_mean, kwargs=kwargs)
        return start_dict
        
    ####### Methods to help with Templates #######    
    def makeTemplate(self, data2D, avg_start_idx=0, avg_stop_idx=None, subtract_mean=True):
        summed_array = np.sum(data2D, axis=0)
        template = summed_array / len(data2D)
        if subtract_mean:
            template = self.subtractMean(np.array([template]), avg_start_idx, avg_stop_idx)[0]
        #mean = np.mean(template[:self.pre_trigger_idx//2])
        #template = template - mean 
        #template = template / (np.max(abs(template)))
        return template 
        
    
    ####### Methods to help with Js making #######
    def fourier2D(self, dt,y):
        """
        works on a 2D array, will take fft of each row independently 
        """
        yf = np.fft.fft(y, axis=-1, norm='forward') # Fourier transform for real-valued y
        f  = np.fft.fftfreq(len(y[0]),dt) # frequencies >= 0
        return f, yf
    
    def periodogram2D(self, dt, y):
        """
        works on a 2D array, will take fft of each row independently 
        """
        f,yf = self.fourier2D(dt,y)
        N = len(f)
        return f, N * dt * np.abs(yf)**2
        
    def makePSD(self, array2D, subtract_mean=True, avg_start_idx=0, avg_stop_idx=None):
        """
        take PSD of each row of array2D, and average
        """
        if subtract_mean:
            array2D = self.subtractMean(array2D, avg_start_idx, avg_stop_idx)
        dt = 1/self.fs
        freqs, stacked_PSDs = self.periodogram2D(dt, array2D)
        #PSD = np.mean(stacked_PSDs, axis=0) 
        stacked_PSDs[:,0] = np.inf # ignore DC component 
        return [stacked_PSDs,  freqs]
   

                                 
    ####### Methods for getting Amplitudes and Chisqs ####### 
    def getAChi2(self, data_array, template, J, force_time_sync=True, subtract_mean=True, avg_start_idx=0, avg_stop_idx=False, plot=True): 
        """
        this works on a 2D array data_array  given in time space. 
        template can be a 1D or 2D array. 
        Takes in a given template and J to apply as an OF for each row of the 2D data_array. 
        returns A (1d array) and chis2 (1d array) from optimal filter generated by template and J.
        the n't element of A and chis2 correspond to the n'th row in the stacked list_of_arrays.
        """ 
        template_shape  = np.shape(template)
        if len(template_shape) == 1: 
            template = np.array([template]) # here we force template to be a 2D array, so fourier2D works with it 
        template = template / (np.max(abs(template)))
        if subtract_mean:
            data_array = self.subtractMean(data_array, avg_start_idx, avg_stop_idx)
            
        freq, template_f = self.fourier2D(1/(self.fs), template)
        template_f = template_f[0] # the [0] makes template_f a 1D array
        freq, data_array_f = self.fourier2D(1/(self.fs), data_array) # here data_array_f is a 2D array. 
        assert len(template_f) == len(data_array_f[0])
        assert len(template_f) == len(J)
        OF = np.conjugate(template_f) / J  
        norm = np.real(np.sum(OF * template_f)) # force imaginaries to 0 (should be 0 anyway)
        As = (np.fft.ifft(OF * data_array_f, axis=-1, norm='forward') / norm)
        a1 = 2 * ( np.sum( np.abs( data_array_f**2 ) / J, axis=-1) ) 
        a2 = 2 * ( np.abs( As )**2 * norm )
        chi2s= (a1 - a2.T).T
        tempAs = np.array([(np.fft.ifft(OF * template_f, axis=-1,norm='forward') / norm ) ]) #all Amplitudes, for all time dealys
        ta1 = 2 * ( np.sum( np.abs( np.array([template_f])**2 ) / J, axis=-1) ) 
        ta2 = 2 * ( np.abs( tempAs )**2 * norm )
        tempchi2s= (ta1 - ta2.T).T
        if force_time_sync == True: inds = np.argmin(tempchi2s, axis=-1) #np.array([multi.pre_trigger_idx]) #
        else: inds  = np.argmin(chi2s, axis=-1) #find index of time that minimzes chi2 
        A = As[:, inds] 
        A=A.real
        chi2 =  chi2s[:, inds]
        tempA = tempAs[:, inds]
        tempA = tempA.real
        count=0
        if plot ==True: 
            fig, ax = self.plotOfInputRecovery(template[0], tempA[0][0], template, freq, J)
            ax[0].set_title("Template")
            fig.tight_layout()
            plt.show()
            for i in range(len(data_array)):
                fig, ax = self.plotOfInputRecovery(data_array[i], A[i], template,freq,  J)
                fig.tight_layout()
                plt.show()
                count +=1
                if count == 1:
                    break
                
        return [A, chi2]
        
    def makeHistAndFit(self, amps, normalize_factor=1,  nbins=100, upscale=1.1, downscale=0.9, maxfev=500000):
        bins=np.linspace(np.min(amps)*downscale, np.max(amps)*upscale, nbins)/normalize_factor
        hist, bin_edges = np.histogram(amps/normalize_factor, bins, density=True)
        bin_centers = (np.roll(bin_edges,1) - (bin_edges))/2
        bin_centers = bin_edges[:-1] - bin_centers[1:]
        
        mean_amp = np.mean(amps)/normalize_factor
        sigma= sum(hist * (bin_centers - mean_amp)**2) /len(amps)
        gauss_params, gauss_cov = curve_fit(self.gaus, bin_centers, hist, p0=[max(hist), mean_amp, sigma], maxfev=maxfev)
        return hist, bin_centers, gauss_params
    
        
    def gaus(self,X,C,X_mean,sigma):
        return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

    ######## PLOTTING FUNCTIONS #######
    
    def plotMany(self,fig, ax,  plot_method,  x, y_list, method_kwargs={'plot_kwargs':{}}):
        first_plot = True
        for e in range(len(y_list)):
            if e > 0: first_plot == False
            method_kwargs['plot_kwargs']['color'] = f'C{e}'
            method_kwargs['first_plot'] = first_plot
            fig, ax= plot_method(fig, ax, x, y_list[e], **method_kwargs)
        return fig, ax
        
    def plotPulseSpecificTemplatesAndPsds(self,  basis, pulse_choices):
        fig, axs = plt.subplots(2)
        for p in pulse_choices:
            template = self.pulse_specific_templates[basis][p]
            x = np.arange(len(template)) / self.fs
            stacked_PSDs, freqs = self.makePSD(np.array([template]))
            psd = np.mean(stacked_PSDs, axis=0)
            fig, ax0 = self.plotTemplate(fig, axs[0],   x, template, basis=basis, plot_kwargs={'color':f'C{p}'})
            fig, ax1 = self.plotPsd(fig, axs[1],  freqs, psd, basis=basis, plot_kwargs={'color':f'C{p}'})
        return fig, [ax0,ax1]

    
    def plotTemplate(self, fig, ax, time_s, template,  basis=None, first_plot=True, label=None, plot_kwargs={}):
        ax.plot(time_s, template,label=label, **plot_kwargs)
        if first_plot == True: 
            ax.set_xlabel('time (s)')
            ax.set_ylabel(basis)
        return fig, ax

    def plotPsd(self, fig, ax, freqs, PSD,  basis=None, first_plot=True, label=None ,plot_kwargs={}):
        ax.loglog(freqs, PSD,label=label, **plot_kwargs)
        if first_plot ==True:
            ax.set_xlabel('freq (Hz)')
            ax.set_ylabel(f'{basis}/sqrt(Hz)')
        return fig, ax 
        
    def plotOfInputRecovery(self, pulse, A, template, freq, J):
        time = np.arange(len(template[0])) / (self.fs) 
        template_f = self.fourier2D(1/(self.fs),template)[1][0]
        pulse_f = self.fourier2D(1/(self.fs), np.array([pulse]))[1][0]
        
        fig, axs = plt.subplots(2)
        axs[0].plot(time, pulse, color='C1',  label='v(t)')
        axs[0].plot(time, A * template[0], color='C2', label='A*s(t)')
        axs[0].set_xlabel('time'); axs[0].set_ylabel('amplitude')
        axs[0].legend()
        
        axs[1].plot(freq, J, label='J')
        axs[1].loglog(freq,np.abs(A * template_f)**2,label=r'$|As^*|^2$')
        axs[1].loglog(freq,np.abs(pulse_f)**2,label=r'$v^{*\dagger}v^*$')
        axs[1].set_xlabel('freq'); axs[1].set_ylabel('PSD')
        axs[1].legend(loc='lower right')
        fig.tight_layout()
        return fig, axs
        
    def plotHist(self,fig, axs, bin_centers, hist, gauss_parameters, templateA,  color='C0', label=None,log=False):
        """
        """
        width = bin_centers[1] - bin_centers[0]
        gauss = self.gaus(bin_centers,*gauss_parameters)
        #print(gauss_parameters)
        residual = hist-gauss
        axs[0].bar(bin_centers, hist, width=width, label=label, alpha=0.5, color=color)
        axs[0].plot(bin_centers , gauss, alpha = 1, color='k')
        axs[0].vlines([templateA], 0, max(hist)*1.5, color=color,linestyle='--', label=f'tempA')
        axs[0].vlines(gauss_parameters[1], ymin=0, ymax=max(hist), color='k', linestyles='dashed',label=f'mean of Gauss {label}')
        axs[1].plot(bin_centers, (residual/gauss)*100, color=color)
        axs[0].set_ylabel('count')
        axs[0].grid("on")
        axs[1].grid("on")
        plt.tight_layout()
        axs[1].set_title(f" Residuals")
        axs[1].set_xlabel("Data: Amplitude");
        axs[0].set_xlabel("Data: Amplitude");
        axs[0].set_title(f" Pulse Amplitudes Histogram and Gaussian Fit")
        axs[0].legend()
        return fig, axs 

    def plotManyHists(self,fig, axs, bin_center_lists, hists_lists, gauss_params_lists, templateA_lists,  kwargs={}):
        for e in range(len(hists_lists)):
            bin_centers = bin_center_lists[e]
            hist = hists_lists[e]
            gauss_params = gaus_params_lists[e]
            templateA = templateA_lists[e]
            fig, axs = self.plotHist(fig, axs, bin_centers, hist, gauss_params, templateA, color=f'C{e}',  *kwargs)
        return fig, axs


    def plotOfOutputs(self, fig, axs, As, chisqs, x_array=None, label=None, title=None,  color='C0'):
        if isinstance(fig, type(None)):
            fig, axs = plt.subplots(2, figsize=(30,10), sharex=True)
        ## TODO: make sure that axs has len at least 2
        if isinstance(x_array, type(None)):
            x_array = np.arange(len(As))
        axs[0].plot(x_array, chisqs, marker='o', markersize=.75, linestyle='', color=color, label=label)
        axs[1].plot(x_array, As, marker='o', markersize=.75, linestyle='', color=color, label=label)
        axs[0].legend(loc='right')
        axs[1].legend(loc='right')
        axs[0].set_ylabel("X^2")
        axs[1].set_ylabel("pulse amp")
        axs[0].set_ylabel("X^2")
        axs[-1].set_xlabel('pulse instance number')
        return fig, axs

    def plotManyOfOutputs(self,fig, axs, As_list, chisqs_list,   kwargs={}):
        for e in range(len(As_list)):
            As = As_list[e]
            if e>0:
                x_array = np.arange(len(As)) + len(As_list[e-1])
            else: x_array = np.arange(len(As))
            chisqs = chisqs_list[e]
            
            fig, axs = self.plotOfOutputs(fig, axs, As, chisqs, x_array=x_array, color=f'C{e}', label=f'idx:{e}', *kwargs)
        return fig, axs

    #def plotSigmaMu(self, gauss_params_list): 

    ####### DELETING FUNCITONS ###########
    def getOriginalIdx(self, idx2remove, removed_file_idxs):
        return idx2remove + np.shape(np.where(np.array(removed_file_idxs) <= idx2remove)[0])[0]
    

    def addRowCutsToDict(self, list_idx, pulse_type, row):
        old_file_idx = self.getOriginalIdx(list_idx, self.file_cuts)
        if old_file_idx not in self.row_cuts.keys():
            self.row_cuts[old_file_idx] = {pulse_type:[]}
        elif pulse_type not in self.row_cuts[old_file_idx].keys():
            self.row_cuts[old_file_idx][pulse_type]= []
        print(type(self.row_cuts[old_file_idx][pulse_type]))
        old_row_idx = self.getOriginalIdx(row, self.row_cuts[old_file_idx][pulse_type])
        self.row_cuts[old_file_idx][pulse_type].append(old_row_idx)
        return

    def populateDicts(self, basis = None):
        if not basis: 
            for basis in self.file_specific_templates.keys():
                self.file_specific_templates = self.populateFileSpecificDict(self.makeTemplate, self.file_specific_templates, basis, subtract_mean=False,  kwargs={})
             
            #for basis in self.file_specific_PSDs.keys():
            #    self.file_specific_PSDs = self.populateFileSpecificDict(self.makePSD, self.file_specific_PSDs, basis, subtract_mean=True,  kwargs={})
               
            for basis in self.pulse_specific_templates.keys():
                self.pulse_specific_templates = self.populatePulseSpecificDict(self.makeTemplate, self.pulse_specific_templates, basis, subtract_mean=False,  kwargs={})
            
            for basis in self.pulse_specific_PSDs.keys():
                self.pulse_specific_PSDs = self.populatePulseSpecificDict(self.makePSD, self.pulse_specific_PSDs, basis, subtract_mean=True,  kwargs={})  
        else: 
            self.file_specific_templates =  self.populateFileSpecificDict(self.makeTemplate, self.file_specific_templates, basis, subtract_mean=True,  kwargs={})
            #self.file_specific_PSDs = self.populateFileSpecificDict(self.makePSD, self.file_specific_PSDs, basis, subtract_mean=True,  kwargs={})
            self.pulse_specific_templates = self.populatePulseSpecificDict(self.makeTemplate, self.pulse_specific_templates , basis, subtract_mean=True,  kwargs={})
            self.pulse_specific_PSDs = self.populatePulseSpecificDict(self.makePSD, self.pulse_specific_PSDs, basis, subtract_mean=False,  kwargs={}) 
        return

        
    def deleteRowFromDebugData(self, pulse_type, row, filename=None, list_idx=None):
        """
        delete row of debugDatAnalsis class
        will need to rerun the pulse sorting from the debugDatAnalysis class
        """
        if (isinstance(filename, type(None))==False  and isinstance(list_idx, type(None))==False):
            print('got too many! doing nothing')
        elif isinstance(list_idx, type(None))==False:
            c = self.debugList[list_idx]
            c.removeRowFromAllSortedAttr(pulse_type=pulse_type, row=row)
            print(f'removing row {row} from file {c.full_filename} pulse type {pulse_type} using debugList idx {list_idx} ')
            self.addRowCutsToDict(list_idx, pulse_type, row)
            self.makeRowMap()
        elif isinstance(filename, type(None))==False: 
            for e, c in enumerate(self.debugList):
                if c.filename == filename:
                    c.removeRowFromAllSortedAttr(pulse_type=pulse_type, row=row)
                    self.addRowCutsToDict(e, pulse_type, row)
                    print(f'removing row {row} from filename {filename} pulse type {pulse_type} using debugList idx {e}')
                    self.makeRowMap()
                    break
        else: print('got nothing, giving nothing')
        return 
        
    def deleteDatAnalysisClass(self, filename=None, list_idx=None):
        """
        delete whole debugAnalsis class
        """
        if (isinstance(filename, type(None))==False  and isinstance(list_idx, type(None))==False):
            print('got too many! doing nothing')
        elif isinstance(list_idx, type(None))==False:
            print(f'removing idx {list_idx}')
            self.debugList.pop(list_idx)
            old_idx = self.getOriginalIdx(list_idx, self.file_cuts)
            self.file_cuts = np.append(self.file_cuts, old_idx)
        elif isinstance(filename, type(None))==False: 
            for e, c in enumerate(self.debugList):
                if c.filename == filename:
                    self.debugList.pop(e)
                    old_idx = self.getOriginalIdx(e, self.file_cuts)
                    self.file_cuts = np.append(self.file_cuts, old_idx)
                    print(f'removing filename {filename}')
                    break
        else: print('got nothing, giving nothing')

    # data gets cut when?
        # given A histogram, some are obviously outliers
        # given chisq, some are obvious outliers 
    # how: want to remove whole datAnalysis classes from input 
    # future how: remove single "pulse events" from a particular datAnalysis class 
    ### end result: histogram, fit, and money histgram plot 

    def getBaselineMeans(self, array2D, avg_start_idx=0, avg_stop_idx=None, subtract_mean=False):
        """
        take row-wise average of a 2D RxC numpyarray, over the columns indicated by the avg_start/stop_idx.
            Parameters:
            -----------
            array2D : 2D ndarray
                Input data to act on.
            avg_start_idx : int, optional. Defaults to 0
                what column to start the array2D row-wise average on
            avg_stop_idx : int, optional. Defaults to None
                what column to stop the array2D row-wise average on
            subtract_mean : bool, optional. Defaults to False
                It doesnt matter what this is! Its not used. 
    
            Returns:
            --------
            means : np array of dim (R,)
                mean of each row over the given colum roi. 
        """
        return np.mean(array2D[:,avg_start_idx:avg_stop_idx], axis=1, keepdims=False)
        
    def getBaselineRMS(self, array2D, avg_start_idx=0, avg_stop_idx=None, subtract_mean=True):
        """
        calculate row-wise RMS of a 2D RxC numpyarray, over the columns indicated by the avg_start/stop_idx.
            Parameters:
            -----------
            array2D : 2D ndarray
                Input data to act on.
            avg_start_idx : int, optional. Defaults to 0
                what column to start the array2D row-wise operation on
            avg_stop_idx : int, optional. Defaults to None
                what column to stop the array2D row-wise operation on
            subtract_mean : bool, optional. Defaults to True
                If True, will subract the mean from each row.
                (hint, for typical RMS you really want this to be true)
    
            Returns:
            --------
            rms : np array of dim (R,)
                RMS of each row over the given colum roi. 
        """
        array2D = array2D[:,avg_start_idx:avg_stop_idx] 
        if subtract_mean:
            array2D = array2D - np.mean(array2D, axis=-1, keepdims=True)
        return np.sqrt(np.mean(array2D**2, axis=-1, keepdims=False))
    def cutArray(self, array2D, avg_start_idx=0, avg_stop_idx=None, subtract_mean=False, rows2cut=[]):
        """
            RxC np 2D array. Will return a the 2D array without the row indices in rows2cut.
            Parameters:
            -----------
            array2D : 2D ndarray
                Input data to act on.
            avg_start_idx : int, optional. Defaults to 0
                DOES NOTHING!!!
            avg_stop_idx : int, optional. Defaults to None
                DOES NOTHING!!!
            subtract_mean : bool, optional. Defaults to False
                DOES NOTHING!!!
            rows2cut : list, optional, Defaults to []
                the indexs of the rows to delete from the array
    
            Returns:
            --------
            2D array without the row indices in rows2cut
        """
        ### TODO: make it so it complains of the idxs in rows2cut dont work for that array
        return np.delete(array2D, obj=rows2cut, axis=0)
    def returnArray(self, array2D, subtract_mean=True, avg_start_idx=0, avg_stop_idx=None):
        if subtract_mean:
            array2D = self.subtractMean(array2D, avg_start_idx, avg_stop_idx)
        return array2D

    def scatter_hist(self, x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
    
        # the scatter plot:
        ax.scatter(x, y)
    
        bins_x = np.linspace(min(x)*1, max(x)*1, 100)#np.arange(-lim, lim + binwidth, binwidth)
        bins_y = np.linspace(min(y)*1, max(y)*1, 100)#np.arange(-lim, lim + binwidth, binwidth)
        print(bins_y[0], bins_y[-1])
        ax_histx.hist(x, bins=bins_x)
        ax_histy.hist(y, bins=bins_y, orientation='horizontal')#bins=bins,
        return ax, ax_histx, ax_histy

    def make2DhistWrapper(self, x_array, y_array, xlabel='', ylabel='' ):
        # Start with a square Figure.
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        ax, ax_histx, ax_histy = self.scatter_hist(x_array, y_array, ax, ax_histx, ax_histy)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax_histx.set_ylabel(ylabel+' counts')
        ax_histy.set_xlabel(ylabel+' counts')
        ax.set_ylim(min(y_array), max(y_array))
        return fig

    def makeStupidColorPlots(self, y_array, chi2, chi2_cut, title):
        fig, axs = plt.subplots(2, sharex=True)
        chi2_real = chi2.T[0]
        y_array = y_array
        x_array = np.arange(len(y_array))
        axs[0].plot(x_array[np.where(chi2_real<chi2_cut)], y_array[np.where(chi2_real<chi2_cut)], color='C2', linestyle='', marker='.')
        axs[0].plot(x_array[np.where(chi2_real>chi2_cut)],y_array[np.where(chi2_real>chi2_cut)], color='C3', linestyle='', marker='.')
        #axs[0].hlines([bl_cut], 0, len(y_array))
    
        axs[1].hlines([chi2_cut], 0, len(y_array))
        axs[1].plot(x_array[np.where(chi2_real<chi2_cut)], chi2_real[np.where(chi2_real<chi2_cut)], color='C2', linestyle='', marker='.')
        axs[1].plot(x_array[np.where(chi2_real>chi2_cut)],chi2_real[np.where(chi2_real>chi2_cut)], color='C3', linestyle='', marker='.')
        axs[1].set_xlabel("pulse instance")
        axs[1].set_ylabel("chi2")
        axs[0].set_ylabel(str(title))
        axs[0].set_title(str(title))
        return fig, axs
    def delete_duplicates_in_list(self, list_of_choice):
        # Use a set to remove duplicates and preserve order
        return list(dict.fromkeys(list_of_choice))
    def counting_err(self, n):
        err_0 = np.sqrt(n)
        #err_0[n<10] = np.sqrt(10)
        err_0[n==0] = 1
        return err_0
    def zfit_gauss(self, data, low_guess, upp_guess, height, binnum,mu_gauss, sigma_gauss,  plotsome=False):
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
            chi2 = sum(((vals - (y_fit*num*(bins[1]-bins[0])))**2)/(self.counting_err(vals)**2))
            
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

    def getBaselineResolution(self, J,template):
        template_f = self.fourier2D( 1/(self.fs),template)[1] 
        DT = len(template[0]) * (1/self.fs)
        integrand = np.abs(template_f[0])**2 / J
        return np.sqrt((DT* np.sum(integrand ))**(-1))
    def spikes_in_region(self, array2D,rms_multiplier, start_idx, stop_idx):
        """
        will return list of row #'s that have a spike in the roi set by start/stop column idxs. 
        spike must be greater than rms_multiplier * rms of that region. rms is calculated over all the rows combined. 
        array2D: 2D np array containing pulse data per row
        start_idx / stop_idx: int, for where to set the region of interest when making cuts
        rms_multiplier: float, when there is a spike above the rms*rms_multiplier, that row idx  will be returned 
        """
        #cuts 1 & 2 are implemented by applying different start/stop_idx 
        rms = self.getBaselineRMS(array2D, subtract_mean=True, avg_start_idx=start_idx,avg_stop_idx=stop_idx)
        #print(rms)
        roi = array2D[:,start_idx:stop_idx] 
        highs = np.max(roi, axis=-1)
        #print(highs)
        #print(rms_multiplier*rms)
        rows2cut = list(np.where(rms_multiplier*rms < highs)[0])
        return rows2cut 
    
    def cut3(self, array2D, pre_trigger_idx, rms_multiplier, max_idx_offset_limit):
        #CUT3: Any window in which the maximum occurs more than 4 samples away from the trigger time
        #is removed from analysis if there exists a point anywhere in the window with an amplitude 
        #greater than 5× the pre-trigger RMS.
        max_idxs = np.argmax(array2D, axis=-1)
        max_offsets = np.abs(max_idxs - pre_trigger_idx)
        maxs = np.max(array2D, axis=-1)
        hidden_pulse_rows = list(np.where(maxs - rms * rms_multiplier < 0)[0])
        cutrows = list(np.where(max_offsets > max_idx_offset_limit)[0])
        rows2cut = list(set(cutrows) - set(hidden_pulse_rows))
        return rows2cut

    def percentileRMS_cut(self, array2D, percentile_top_cut, percentile_bot_cut, start_idx, stop_idx):
        rms = self.getBaselineRMS(array2D, subtract_mean=True, avg_start_idx=start_idx,avg_stop_idx=stop_idx)
        roi = array2D[:,start_idx:stop_idx] 
        top_rms_percentile = np.percentile(rms, percentile_top_cut, axis=-1) 
        bot_rms_percentile = np.percentile(rms, percentile_bot_cut, axis=-1) 
        rows2cut = np.where((rms < bot_rms_percentile) | (rms > top_rms_percentile))[0]
        return rows2cut 
    def percentileMEAN_cut(self, array2D, percentile_top_cut, percentile_bot_cut, start_idx, stop_idx):
        means = self.getBaselineMeans(array2D, subtract_mean=False, avg_start_idx=start_idx,avg_stop_idx=stop_idx)
        roi = array2D[:,start_idx:stop_idx] 
        top_mean_percentile = np.percentile(means, percentile_top_cut, axis=-1) 
        bot_mean_percentile = np.percentile(means, percentile_bot_cut, axis=-1) 
        rows2cut = np.where((means < bot_mean_percentile) | (means > top_mean_percentile))[0]
        return rows2cut 


        




