import numpy as np
import neurokit2 as nk
from scipy.interpolate import CubicSpline
from scipy.signal import cheby2, filtfilt,butter
from scipy.signal import resample, savgol_filter

class temp_find:
    ppg_fre = 60
    num = 128

    def __init__(self,ppg = None):
        self.ppg = ppg
        
    def temping(self):
        # temp = []
        # no_dc = self.dc_take(self.ppg)
        # tem = (-1 * np.array(no_dc))
        # tem = (tem-np.min(tem))/(np.max(tem)-np.min(tem))
        # ppg_peaks = self.ppg_peak(tem)
        # no_base = no_dc-self.spline(ppg_peaks, no_dc)
        # for i in range(len(ppg_peaks)-1):
        #     a = self.liesample(no_base[ppg_peaks[int(i)]:ppg_peaks[int(i+1)]],self.num) 
        #     temp.append(a)
        # return ppg_peaks,temp
        fs = self.ppg_fre  # Sampling rate
        N = len(self.ppg)
        t = np.arange(N) / fs
        s = np.array(self.ppg)

        # --------------------------------
    
        # --------------------------------
        lowcut = 0.5
        highcut = 3
        order = 4

        b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
        filtered = filtfilt(b, a, s)
        s1 = filtered

        _, results = nk.ppg_peaks(s, sampling_rate=fs)
        peak_idx = results["PPG_Peaks"]
        peak_val = s[peak_idx]

        valley_idx = []
        valley_val = []

        for i in range(len(peak_idx) - 1):
            start = peak_idx[i]
            end = peak_idx[i + 1]

            # segment between two peaks
            segment = s[start:end]

            # find local minimum in that segment
            local_min_index = np.argmin(segment)

            # convert to global index
            global_min_index = start + local_min_index

            valley_idx.append(global_min_index)
            valley_val.append(s[global_min_index])

        valley_idx = np.array(valley_idx)
        temp = []
        no_dc = self.dc_take(s)
        no_base = no_dc-self.spline(valley_idx, no_dc)
        for i in range(len(valley_idx)-1):
            a = self.liesample(no_base[valley_idx[int(i)]:valley_idx[int(i+1)]],self.num) 
            temp.append(a)
        return valley_idx,temp
        
        return valley_idx,valley_val

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

        b, a = cheby2(order, ripple, [low_norm, high_norm], btype='band')

        return filtfilt(b, a, raw_signal)
    
    def spline(self,peak,ppg):
        peak_val = [ppg[i] for i in peak]
        # create spline
        cs = CubicSpline(peak, peak_val)

        # smooth x values
        x_full = np.arange(len(ppg))
        return cs(x_full)
    
    def liesample(self,signal, num_points):
        y_smooth = savgol_filter(signal, window_length=11, polyorder=3)
        resam = resample(y_smooth, num_points)
        a = (resam[-1]-resam[0])/(len(resam)-1)
        b = resam[0]
        c = a*np.arange(len(resam))+b

        return resam-c 
#Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    with open("D:/my_project/my_tool/ppg_noise_reject/utils/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    a = temp_find()
    a.ppg = data["PPG"]
    no_dc = a.dc_take(a.ppg)
    peaks,temp = a.temping()
    spline = a.spline(peaks, no_dc)
    plt.plot(no_dc)
    plt.plot(spline)
    plt.show()
    plt.plot(no_dc-spline)
    plt.show()
