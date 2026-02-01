import numpy as np
import neurokit2 as nk
from scipy.signal import cheby2, filtfilt

class temp_find:
    ppg_fre = 60

    def __init__(self,ppg = None):
        self.ppg = ppg
        
    def temping(self):
        temp = []
        tem = (-1 * np.array(self.ppg))
        print(tem)
        tem = (tem-np.min(tem))/(np.max(tem)-np.min(tem))
        ppg_peaks = self.ppg_peak(tem)
        no_dc = self.dc_take(self.ppg)
        for i in range(len(ppg_peaks)-1):
            temp.append(no_dc[ppg_peaks[int(i)]:ppg_peaks[int(i+1)]])
        return ppg_peaks,temp

    def ppg_peak(self,ppg):
        _, results = nk.ppg_peaks(ppg, sampling_rate = self.ppg_fre)
        return results["PPG_Peaks"]
    
    def dc_take(self,raw_signal):
        fs = self.ppg_fre# <-- CHANGE THIS to your real sampling rate
        low = 0.03
        high = 5

        # Normalize frequencies (required by scipy)
        low_norm = low / (fs / 2)
        high_norm = high / (fs / 2)

        order = 1         # filter order
        ripple = 0.1      # dB ripple in passband


        # Design Chebyshev Type II low-pass filter
        b, a = cheby2(order, ripple, [low_norm, high_norm], btype='band')

        return filtfilt(b, a, raw_signal)
    
#Test
if __name__ == "__main__":
    a = temp_find()
    