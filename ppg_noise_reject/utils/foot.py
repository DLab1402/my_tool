from PyEMD import EMD
import numpy as np
import pylab as plt
import os
import random
import json
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import argrelextrema


path = "D:/my_project/valid_data_8_2_26_resample"
files = [f for f in os.listdir(path) if f.endswith(".json")]

n = len(files)
idx = random.randint(0, n - 1)
file_path = os.path.join(path, files[idx])

with open(file_path, "r") as f:
    data = json.load(f)

def main(s):
    fs = 60  # Sampling rate
    N = len(s)
    t = np.arange(N) / fs

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
    valley_idx = t[valley_idx]  # Exclude first and last valleys
    valley_val = np.array(valley_val)
    threshold = 0.5

    # Allocate gradients
    left_grad  = np.zeros(len(valley_val))
    right_grad = np.zeros(len(valley_val))

    # Compute gradients
    for i in range(1, len(valley_val) - 1):
        left_grad[i] = valley_idx[i] - valley_idx[i-1]#(valley_val[i] - valley_val[i-1]) / (valley_idx[i] - valley_idx[i-1])
        right_grad[i] = valley_idx[i+1] - valley_idx[i]#(valley_val[i+1] - valley_val[i]) / (valley_idx[i+1] - valley_idx[i])

    # edges cannot compute both sides
    left_grad[0] = np.nan
    right_grad[-1] = np.nan

    left_abs = np.abs(left_grad[~np.isnan(left_grad)])



    # --------------------------------------------------
    # Reject condition
    # reject_mask = (
    #     (left_grad  > 0) &
    #     (right_grad < 0) &
    #     (np.abs(left_grad)  > threshold) &
    #     (np.abs(right_grad) > threshold)
    # )

    reject_mask = (
        (np.abs(left_grad)  < threshold) |
        (np.abs(right_grad) < threshold)
    )

    # Keep valid samples
    keep_mask = ~reject_mask

    final_idx   = valley_idx[keep_mask]
    final_value = valley_val[keep_mask]

    print("Rejected:", np.where(reject_mask)[0])
    print("Kept:", np.where(keep_mask)[0])

    # --------------------------------
    # 3️⃣ FFT
    # --------------------------------
    filtered = filtered - np.mean(filtered)
    filtered = filtered * np.hanning(N)

    fft_vals = np.fft.fft(filtered)
    fft_freq = np.fft.fftfreq(N, 1/fs)

    positive = fft_freq > 0
    fft_vals = fft_vals[positive]
    fft_freq = fft_freq[positive]
    fft_amp = np.abs(fft_vals) / N

    # --------------------------------
    # 4️⃣ Select 0.5–3 Hz band
    # --------------------------------
    band = (fft_freq >= 0.5) & (fft_freq <= 3)

    freq_band = fft_freq[band]
    amp_band = fft_amp[band]

    # --------------------------------
    # 5️⃣ Find Top 3 Peaks in Band
    # --------------------------------
    peaks_fft, _ = find_peaks(amp_band)

    peak_freqs = freq_band[peaks_fft]
    peak_amps = amp_band[peaks_fft]

    sorted_idx = np.argsort(peak_amps)[::-1]

    top3_freqs = peak_freqs[sorted_idx[:3]]
    top3_amps = peak_amps[sorted_idx[:3]]

    print("Top 3 FFT Peaks (0.5–3 Hz):")
    for i in range(len(top3_freqs)):
        print(f"{i+1}: {top3_freqs[i]:.3f} Hz ({top3_freqs[i]*60:.1f} BPM)")

    # --------------------------------
    # 6️⃣ Plot
    # --------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))

    ax[0].plot(t, s, "g", label="Raw Signal")
    ax[0].plot(t[peak_idx], peak_val, "bo", label="Detected Peaks")
    ax[0].plot(valley_idx, valley_val, "ko", label="Detected Feet")
    ax[0].plot(final_idx, final_value, "ro", label="Accepted Feet")
    ax[0].set_title("PPG Signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].legend()

    # FFT
    ax[1].plot(freq_band, amp_band, "b")
    ax[1].scatter(top3_freqs, top3_amps)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("FFT (0.5–3 Hz)")

    ax[2].hist(left_abs, bins=15)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(np.array(data["Syn_PPG"]))