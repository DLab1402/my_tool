import numpy as np
import neurokit2 as nk
from scipy.signal import cheby2, filtfilt

class temp_find:
    ppg_fre = 60

    def __init__(self,ppg = None):
        self.ppg = ppg
        
    def temping(self):
        temp = []
        tem = (tem-np.min(tem))/(np.max(tem)-np.min(tem))
        ppg_peaks = self.ppg_peak(tem)
        no_dc = self.dc_take(self.ppg)
        for i in range(len(ppg_peaks)-1):
            temp.append(no_dc[ppg_peaks[i]:ppg_peaks[i+1]])
        return temp

    def ppg_peak(self,ppg):
        _, results = nk.ppg_peaks(ppg, sampling_rate = self.ppg_fre)
        ppg_peaks = np.zeros(len(ppg))
        return ppg_peaks
    
    def dc_take(self,raw_signal):
        # Sampling parameters
        fs = self.ppg_fre        # sampling frequency (Hz)
        cutoff = 5      # cutoff frequency (Hz)
        order = 4       # filter order
        rs = 40         # stopband attenuation (dB)

        # Normalize cutoff frequency
        wn = cutoff / (fs / 2)

        # Design Chebyshev Type II low-pass filter
        b, a = cheby2(order, rs, wn, btype='low', analog=False)

        return filtfilt(b, a, raw_signal)
    
#Test
if __name__ == "__main__":
    a = temp_find()
    