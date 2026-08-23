import json
import numpy as np 

def getFileDict(filename, log_file,):
    """
    This takes in the name of a file. It will find the corresponding entry in the log file and then
    returns a dictionary which will include:
    - band: band data was taken on.
    - channels: channels data was taken on. 
    - filename: the filename of the data.
    - nsamp: the number of samples collected.
    - freq_in_Hz: the tone of the readout frequency.
    - eta_phase_radian: ---ask
    - fs: the sample rate
    - att_uc: attenuation
    - att_dc: dc attenuation
    - period_us: the period of the pulse sequence in us
    - tunefile: the tune file
    - awg_settings: a dictionary of moku settings with
        - power
        ??????????????????
    -led_settings:
    -MEMS: the MEMS settings
        """
    read_log = open(log_file, 'r')
    file_search= filename.split('/')[-1]
    file_search = file_search.split('.')[0]
    done=False
    found=False
    text_dict=''
    print("JSON DICT FOR " + file_search)
    for l_no, line in enumerate(read_log):
        if found and not done:
            text_dict=line
            done = True
        elif "JSON DICT FOR " + file_search in line:
                found=True
    startIndex = text_dict.find("{")
    file_dict = json.loads(text_dict[startIndex:])
        
    return file_dict

def getFilesFromLog(log_file):
    """
    loads all file names and times following the "JSON DICT FOR " in the logfile
        """
    ftime_list = []
    fname_list = []
    read_log = open(log_file, 'r')
    text_flag = "JSON DICT FOR "
    for l_no, line in enumerate(read_log):
        
        if text_flag in line:
            fname = line.split(' ')[-1]
            ftime = fname.split('_')[0]
            ftime_list.append(ftime) 
            fname_list.append(fname[:-1]+'.dat') 
    return ftime_list, fname_list

def findFilesBetweenPairs(full_file_list, file_pair):
    """
    Returns the indices of two files within a list.
    
    Parameters:
    ----------
    full_file_list : list
        List of file names.
    file_pair : tuple
        A tuple with two file names (start, end).

    Returns:
    -------
    tuple
        Indices of start and end files.

    Example:
    --------
    findFilesBetweenPairs(["file1", "file2", "file3"], ("file1", "file3"))
    # Returns: 0, 2
    """
    full_file_array = np.array(full_file_list)
    start_idx = int(np.where(full_file_array == file_pair[0])[0])
    stop_idx = int(np.where(full_file_array == file_pair[1])[0])
    return start_idx, stop_idx


