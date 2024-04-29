#EMG FILTER 
import scipy.signal
from scipy import signal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo


class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))



#PARAMETER for FILTER
pole = 5
lowcut = 10.0
highcut = 80.0
samp_freq = 200
notch_freq = 50
quality_factor = 90

#PARAMETER for SLIDING WINDOW
window = 40
step = 5


#FILTER DEFINITION
import scipy
from scipy.signal import butter, lfilter
from scipy import signal
def filters (chanel,pole,low,high,fs):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs=fs)
    dn = signal.filtfilt(b_notch, a_notch, chanel,axis=0)
    b, a = scipy.signal.butter(pole, [low, high], 'band', fs=fs)
    df = scipy.signal.lfilter(b, a, dn,axis=0)
    dframe = pd.DataFrame(df)
    return dframe

#SLIDING WINDOW DEFINITION + EXTRACTION FEATURE
from numpy.lib.stride_tricks import as_strided
import math
from scipy import stats
def moving_window_stride(array, window, step):
   
    stride = array.strides[0]
    win_count = math.floor((len(array) - window + step) / step)
    strided = as_strided(array, shape=(win_count, window), strides=(stride*step, stride))
    index = np.arange(window - 1, window + (win_count-1) * step, step)
    return strided, index

def feature_rms(series, window, step):
    """Root Mean Square"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(windows_strided), axis=1)), index=series.index[indexes]) 
 
#EXECUTE ALL DEFINITION OF PROCESSING IN "PREP"
def prep(dat_emg):
    
    datafilter1 = filters(dat_emg,pole,lowcut, highcut,fs=samp_freq)
    datafilter = pd.DataFrame(datafilter1)
    datafilter.columns = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    bp = pd.DataFrame(datafilter)
    bp.columns = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    X_RMS = []
    for i in range(bp.shape[1]):
        X_RMS.append(feature_rms(bp.iloc[:, i], window, step))
    X_RMS = pd.DataFrame(np.asarray(X_RMS).T)
    
    return X_RMS
