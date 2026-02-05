import numpy as np
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pylab as plt

def chunk_data(fs_MHz, chunk_duration_us, data,  start=0):
    """
    fs_MHz: sample freq, MHz
    chunk_duration_us: period of wave on LED 
    data: np array with shape: (n_samp,) data to chunk up 
    start: idx to start your chunks on, so you can center pulses in the window 
    returns: raw data, separated into chunks. 
    """
    n_samp = len(data)
    # Calculate the number of samples per chunk
    samples_per_chunk = int(fs_MHz * chunk_duration_us)
    num_chunks = n_samp // samples_per_chunk 
    if start != 0:
        num_chunks -= 1
    #remove the end of the data, so you can have divisible chunks
    data_chunks = data[start:num_chunks*samples_per_chunk+start]

    # Reshape the array into chunks
    data_chunks = data_chunks.reshape(num_chunks, (samples_per_chunk))

    return data_chunks



def subtractTrendAndRecenter(chunked_data, time_chunk, num_pulse_types=1,pulse_choice=0, pulses_to_fit_shift=np.array([]), pulse_loc_scale=1):
    """
    chunked data: np array, [nchunks x nsamp per chunks] 
    time_chunk: time array for one single chunk
    num_pulse_types: (int) number of pulses sent in (used for sawtooth x square wave) assumes pulse amplitude repeats at this rate
    pulse_choice: (int) which pulse to use for your average... (used for sawtooth x square wave) must be < num_pulse_types
    pulses_to_fit_shift: (iterable) which pulses to include in your moku/smurf shift, numbers must be < num_pulse_types 
    """
    data_for_fit = chunked_data.imag #Just use Q quadrature for determining where mins should be 
    
    if len(pulses_to_fit_shift) == 0: 
        pulses_to_fit_shift = np.arange(num_pulse_types)
        print("here")
    
    # I want to use only i'th chunk in the min indx...
    num_chunks = len(chunked_data)
    chunk_len = len(chunked_data[0])
    choice_array_fit = np.arange(num_chunks*num_pulse_types).reshape(num_chunks,num_pulse_types).T  
    mask_fit = np.ma.masked_greater_equal(choice_array_fit, num_chunks)
    choices = mask_fit[pulses_to_fit_shift].compressed()   
    chunked_data_for_fit = data_for_fit[choices] #only has chunks for pules you want to use to determine your shift
    
    x_array = np.arange(num_chunks)[choices] #nchunks, but only ones corresponding to pulses we are using
    min_idx_data = []  # will be populated with idx of minimum vaule of each chunk
    for i in range(len(chunked_data_for_fit)): 
        index = np.argmin(chunked_data_for_fit[i,:])
        min_idx_data.append(index)
    min_idx_data = np.array(min_idx_data) #if not using all pulses, this is SHORTER than the total num_chunks
    print(f'total chunks: {num_chunks},number of chunks used in fit:{len(min_idx_data)}') 
    
    p_fit= np.polyfit(x_array, np.array(min_idx_data), 1) #these are the params for the fit
    print(p_fit)
    fit = np.poly1d(p_fit) #this is a function, seeded with the output of the fit
    center_idx = len(time_chunk)//2  #index of center time... gotta pick something!
    print("center_idx:", center_idx)
    
    full_x_array = np.arange(num_chunks)
    plt.plot(x_array, np.array(min_idx_data)-center_idx, label='mins')
    plt.plot(full_x_array, fit(full_x_array)-center_idx, label='fit')
    plt.legend(loc='best')
    plt.xlabel('chunk')
    plt.ylabel('idx shift to center')
    plt.show()

    
    idx_to_shift = fit(full_x_array) - int(pulse_loc_scale*center_idx )
    max_shift = int(min(idx_to_shift))  ###assumes pulses are left of center!!!!
    print('max_shift:',max_shift)
    
    time_crop = time_chunk[-max_shift :max_shift]
    print(len(time_crop))
    data_holder = np.zeros([len(chunked_data), len(time_crop)], dtype=complex)
    fig, ax = plt.subplots(2)
    for i, shift in enumerate(idx_to_shift):
        shift_to_max = int(shift)
        remaining_shift = int(shift_to_max-max_shift)
        chunk = chunked_data[i,:]
        
        if remaining_shift != 0:
            chunk_cropped =chunk[remaining_shift:shift_to_max+max_shift]
            
        else: chunk_cropped =chunk[:shift_to_max+max_shift]
        
        data_holder[i] = chunk_cropped
        if i<10: 
            ax[0].plot(time_crop, chunk_cropped.imag, alpha=.5)
            ax[1].plot(time_crop, chunk_cropped.real, alpha=.5)
            
        if i > 1390: 
            ax[0].plot(time_crop, chunk_cropped.imag, alpha=1)
            ax[1].plot(time_crop, chunk_cropped.real, alpha=1)

            

    #average over specific chunks: 
    choice_array = np.arange(num_chunks*num_pulse_types).reshape(num_chunks,num_pulse_types).T
    mask = np.ma.masked_greater_equal(choice_array, num_chunks)
    choices = mask[pulse_choice].compressed()
    average_chunks_real = np.sum(data_holder.real[choices], 0)/len(choices)
    average_chunks_imag = np.sum(data_holder.imag[choices], 0)/len(choices)
    average_chunks  = average_chunks_real + 1j*average_chunks_imag
    ax[0].plot(time_crop, average_chunks_imag, alpha=1)
    ax[1].plot(time_crop, average_chunks.real, alpha=1)
    plt.show()
    return data_holder, average_chunks, time_crop


def fourier(y,dt, allreal=True):
    if allreal:
        yf = np.fft.rfft(y) # Fourier transform for real-valued y
        f  = np.fft.rfftfreq(len(y),dt) # frequencies >= 0
    else:
        yf = np.fft.fft(y)
        f  = np.fft.fftfreq(len(y),dt)
        yf = yf[f>=0] # frequencies >= 0
        f = f[f>=0]
    return f, yf


def periodogram(y,dt, allreal=True):
    f,yf = fourier(y,dt, allreal)
    return f, np.abs(yf)**2


def pulse_choices(num_pulse_types, num_chunks, chunk_len, pulse_choice):
    """
    num_pulse_types: (int) number of pulses sent in (used for sawtooth x square wave) assumes pulse amplitude repeats at this rate
    num_chunks: (int) number of chunks you have split your data into 
    chunk_len: (int) length of each chunk 
    pulse_choice: (int) which pulse to look at... (used for sawtooth x square wave) must be < num_pulse_types 

    returns: choices, an array of indices that correspond to the pulses of choice.
    """
    choice_array_fit = np.arange(num_chunks*num_pulse_types).reshape(num_chunks,num_pulse_types).T  
    mask_fit = np.ma.masked_greater_equal(choice_array_fit, num_chunks)
    choices = mask_fit[pulse_choice].compressed()   
    return choices

