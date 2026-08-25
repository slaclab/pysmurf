import numpy as np
class FourierFunctions:
    def __init__(self):
        return
    def fourier2D(self, dt,y):
        """
        works on a 2D array, will take fft of each row independently 
        """
        yf = np.fft.fft(y, axis=-1, norm='forward') 
        f  = np.fft.fftfreq(len(y[0]),dt) 
        return f, yf
    
    def periodogram2D(self, dt, y):
        """
        works on a 2D array, will take fft of each row independently 
        """
        f,yf = self.fourier2D(dt,y)
        N = len(f)
        return f, N * dt * np.abs(yf)**2
    
    def makePSD(self, array2D,fs, subtract_mean=True, avg_start_idx=0, avg_stop_idx=None):
        """
        take PSD of each row of array2D, and average
        """
        if subtract_mean:
            array2D = self.subtractMean(array2D, avg_start_idx, avg_stop_idx)
        dt = 1/fs
        freqs, stacked_PSDs = self.periodogram2D(dt, array2D)
        #PSD = np.mean(stacked_PSDs, axis=0) 
        stacked_PSDs[:,0] = np.inf # ignore DC component 
        return [stacked_PSDs,  freqs]
    
    def subtractMean(self, array2D, start_idx, stop_idx):
        return  array2D - np.mean( array2D[:,start_idx:stop_idx], axis=1, keepdims=True)
                      

