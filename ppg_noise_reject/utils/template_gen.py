import numpy as np
import neurokit2 as nk
from scipy.interpolate import CubicSpline
from scipy.signal import cheby2, filtfilt,butter
from scipy.signal import resample, savgol_filter
from sklearn.ensemble import IsolationForest

class temp_find:
    ppg_fre = 60
    num = 128

    def __init__(self,ppg = None):
        self.ppg = ppg
        
    def temping(self):
        fs = self.ppg_fre  # Sampling rate
        N = len(self.ppg)
        t = np.arange(N) / fs
        s = np.array(self.ppg)

        lowcut = 0.5
        highcut = 3
        order = 4

        b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
        filtered = filtfilt(b, a, s)

        _, results = nk.ppg_peaks(filtered, sampling_rate=fs)
        peak_idx = results["PPG_Peaks"]

        # peak_idx = np.array(peak_idx)
        # peak_val = [filtered[i] for i in peak_idx]

        # peak_val = np.array(peak_val)


        # interval = np.diff(peak_idx)
        # # plt.plot(interval)
        # # plt.show()
        # valid = peak_idx.copy()  # copy to mark valid peaks

        # for i in range(len(peak_idx)-1):
        #     start = peak_idx[i]
        #     end = peak_idx[i + 1]

        #     # segment between two peaks
        #     segment = filtered[start:end]

            
        #     # find local minimum in that segment
        #     local_min_index = np.argmin(segment)
        #     h1 = peak_val[i] - segment[local_min_index]
        #     h2 = peak_val[i+1] - segment[local_min_index]
            
        #     if interval[i] < 0.4*fs:  # less than 0.5 seconds                    
        #         if peak_val[i] > peak_val[i+1]:
        #             valid[i+1] = -1  # mark for removal
            
        #     if fs > interval[i] and interval[i] >= 0.4*fs:  # greater than 2 seconds
        #         if h2/h1 < 0.7:  # if the next peak is not significantly higher than the current one
        #             valid[i+1] = -1  # mark for removal
        #         # else:
        #         #     valid[i] = -1  # mark for removal
            

        # # Remove marked peaks        peak_idx = peak_idx[peak_idx != -1]
        # peak_val = [filtered[i] for i in valid if i != -1]
        # # print("peak_val:", peak_val)
        # peak_idx = np.array([i for i in valid if i != -1])
        # # X = peak_val.reshape(-1, 1)

        # # clf = IsolationForest(contamination=0.05)  # tune this
        # # labels = clf.fit_predict(X)

        # # # keep normal points (label = 1)
        # # mask = np.where(labels == 1)

        # # peak_val = peak_val[mask]
        # # peak_idx = peak_idx[mask]  # keep index aligned

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
            valley_val.append(filtered[global_min_index])

        valley_idx = np.array(valley_idx)
        valley_val = np.array(valley_val)

        # # Compute gradient around each valley
        # grad_left = []
        # grad_right = []

        # for i, idx in enumerate(valley_idx[1:-1]):
        #     x1 = valley_idx[i-1]  # previous valley
        #     x2 = valley_idx[i + 1]  # next valley
        #     y1 = valley_val[i-1]  # previous valley value
        #     y2 = valley_val[i + 1]  # next valley value
        #     g_left = (valley_val[i] - y1) / (idx - x1)
        #     g_right = (y2 - valley_val[i]) / (x2 - idx)

        #     grad_left.append(g_left)
        #     grad_right.append(g_right)

        # grad_left = np.array(grad_left)
        # grad_right = np.array(grad_right)

        # # print("grad_left:", grad_left)
        # # print("grad_right:", grad_right)

        # threshold = 0.00001  # tune this based on your signal scale

        # valid_indices = []

        # for i in range(len(valley_idx[1:-1])):
        #     g_left = grad_left[i]
        #     g_right = grad_right[i]

        #     # Reject abnormal valleys
        #     if (g_left > threshold) and (g_right > -threshold):
        #         continue  # skip this one (eject)

        #     valid_indices.append(i)
        
        # valley_idx = valley_idx[valid_indices]


        temp = []
        no_dc = self.dc_take(s)
        no_base = no_dc-self.spline(valley_idx, no_dc)
        for i in range(len(valley_idx)-1):
            a = self.liesample(no_base[valley_idx[int(i)]:valley_idx[int(i+1)]],self.num) 
            temp.append(a)

        
        return peak_idx,valley_idx,temp
        

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
        a = (signal[-1]-signal[0])/(len(signal)-1)
        b = signal[-1]
        c = a*np.arange(len(signal))+b
        y = signal - c
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)
        resam = resample(y_smooth, num_points)
        return resam
#Test
if __name__ == "__main__":
    import pylab as plt
    import os
    import random
    import json
    from matplotlib.patches import Rectangle
    plt.style.use('dark_background')
    # path = "D:/my_project/valid_data_8_2_26_resample"
    path = "G:/My Drive/valid_data_8_2_26_resample2"
    files = [f for f in os.listdir(path) if f.endswith(".json")]

    n = len(files)
    idx = random.randint(0, n - 1)
    # idx = 127
    file_path = os.path.join(path, files[idx])

    a = temp_find()

    with open(file_path, "r") as f:
        data = json.load(f)
    
    a.ppg = data["PPG"]
    no_dc = a.dc_take(a.ppg)
    peaks,feet,temp = a.temping()
    foot = random.randint(0, len(feet) - 1)
    foot = 12+48
    print("foot:", foot)
    print("idx:", idx)
    f1 = feet[foot]
    f2 = feet[foot+1]
    spline = a.spline(feet, no_dc)

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    ax[0].plot(a.ppg)
    ax[0].set_xlim(500, 8000)
    ax[0].grid()
    ax[0].set_title("(A)")

    ax[1].plot(no_dc)
    ax[1].plot(peaks, no_dc[peaks], "ro")
    ax[1].plot(feet, no_dc[feet], "ko")
    ax[1].plot(spline)
    ax[1].set_xlim(500, 8000)
    ax[1].set_ylim(-100, 100)
    ax[1].grid()
    ax[1].set_title("(B)")

    final = no_dc - spline
    ax[2].plot(final)
    ax[2].set_xlim(500, 8000)
    ax[2].set_ylim(-25, 110)
    ax[2].grid()
    ax[2].set_title("(C)")
    valid_rec = Rectangle((f1, -100), f2-f1, 225, facecolor='red', edgecolor='none', alpha=0.3)
    ax[2].add_patch(valid_rec)
    # plt.draw()
    plt.show()

    # tp = no_dc[f1:f2]
    # plt.plot(tp)
    # plt.grid()
    # plt.show()

    # tp1 = a.liesample(final[f1:f2], a.num)
    # tp1 = (tp1-np.min(tp1))/(np.max(tp1)-np.min(tp1))
    # plt.plot(tp1)
    # plt.grid()
    # plt.show()


    # tp2 = no_dc[f1:f2]
    # tp2 = a.liesample(tp2, a.num)
    # a = (tp2-tp2[0])/(len(tp2)-1)
    # b = tp2[-1]
    # c = a*np.arange(len(tp2))+b
    # tp2 = tp2 - c
    # tp2 = (tp2-np.min(tp2))/(np.max(tp2)-np.min(tp2))
    # plt.plot(tp2)
    # plt.grid()
    # plt.show()

    # 182 113, 42 82, 151 93, 144 135, 9 90, 105 86, 139 9, 109 40