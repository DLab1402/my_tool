import os
import json
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

ECG_FS = 1000
PPG_FS = 60

PATH = r"H:\My Drive\data_set_review\data"
SAVE_PATH = "H:\\My Drive\\data_set_review\\data_notem"

file_list = [f for f in os.listdir(PATH) if f.endswith(".json")]


def normalize(sig):
    sig = np.array(sig, dtype=float)

    rng = sig.max() - sig.min()

    if rng == 0:
        return sig

    return (sig - sig.min()) / rng


def downsample_signal(sig, fs_in, fs_out):

    from math import gcd

    g = gcd(fs_in, fs_out)

    up = fs_out // g
    down = fs_in // g

    return resample_poly(sig, up, down)


def detect_ecg_rpeaks(ecg):

    signals, info = nk.ecg_process(ecg, sampling_rate=ECG_FS)

    return info["ECG_R_Peaks"]


def detect_ppg_peaks(ppg):

    signals, info = nk.ppg_process(ppg, sampling_rate=PPG_FS)

    return info["PPG_Peaks"]


def build_impulse_train(peaks, length):

    impulse = np.zeros(length)

    peaks = peaks[peaks < length]

    impulse[peaks] = 1

    return impulse


def compute_lag(ecg_peaks_ds, ppg_peaks, signal_len):

    ecg_impulse = build_impulse_train(ecg_peaks_ds, signal_len)

    ppg_impulse = build_impulse_train(ppg_peaks, signal_len)

    cc = np.correlate(ppg_impulse, ecg_impulse, mode='full')

    lag = np.argmax(cc) - (len(ecg_impulse) - 1)

    return lag


def shift_signal(sig, lag, target_len):

    out = np.zeros(target_len)

    if lag > 0:

        n = min(len(sig), target_len - lag)

        out[lag:lag+n] = sig[:n]

    elif lag < 0:

        lag = abs(lag)

        n = min(len(sig) - lag, target_len)

        out[:n] = sig[lag:lag+n]

    else:

        n = min(len(sig), target_len)

        out[:n] = sig[:n]

    return out


for file in file_list:

    try:

        file_path = os.path.join(PATH, file)

        with open(file_path, 'r') as f:
            data = json.load(f)

        ecg = np.array(data["ECG"], dtype=float)
        ppg = np.array(data["PPG"], dtype=float)
        label = np.array(data["Label"], dtype=float)

        # normalize
        ppg = normalize(ppg)

        # downsample ECG + label
        ecg_ds = downsample_signal(ecg, ECG_FS, PPG_FS)
        label_ds = downsample_signal(label, ECG_FS, PPG_FS)

        ecg_ds = normalize(ecg_ds)

        # detect ECG R-peaks
        rpeaks = detect_ecg_rpeaks(ecg)

        # convert ECG peak indices to PPG sampling domain
        ratio = PPG_FS / ECG_FS

        rpeaks_ds = np.round(rpeaks * ratio).astype(int)

        # detect PPG peaks
        ppg_peaks = detect_ppg_peaks(ppg)

        # compute lag
        lag = compute_lag(
            rpeaks_ds,
            ppg_peaks,
            min(len(ecg_ds), len(ppg))
        )

        # align ECG
        ecg_aligned = shift_signal(
            ecg_ds,
            lag,
            len(ppg)
        )

        # trim
        min_len = min(
            len(ppg),
            len(ecg_aligned),
            len(label_ds)
        )

        ppg = ppg[:min_len]
        ecg_aligned = ecg_aligned[:min_len]
        label_ds = label_ds[:min_len]

        raw_data = {
        "PPG": ppg.tolist(),
        "ECG": ecg_aligned.tolist(),
        "Label": label_ds.tolist()
        }
        raw_file_path = os.path.join(SAVE_PATH, file)
        with open(raw_file_path, 'w') as f:
            json.dump(raw_data, f)

        # plot
        # plt.figure(figsize=(15, 5))

        # plt.plot(ppg, label="PPG")
        # plt.plot(ecg_aligned, label="ECG aligned")
        # plt.plot(label_ds, label="Label")

        # plt.legend()
        # plt.title(file)

        # plt.show()

    except Exception as e:

        print(f"Error processing {file}: {e}")