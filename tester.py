import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt
from pan_tompkins_qrs import Pan_Tompkins_QRS
from wqrsm import wqrsm
import bsqi

def initialize_HRVparams(fs):
    """
    Initializes HRV parameters for SQI computation.
    """
    return {
        'sqi': {
            'windowlength': 30,
            'TimeThreshold': 0.1,
            'margin': 0.05,
            'increment': 30  # window length and increment should be the same
        },
        'Fs': fs,
        'PeakDetect': {
            'REF_PERIOD': 0.250,
            'THRES': 0.6,
            'fid_vec': [],
            'SIGN_FORCE': [],
            'debug': True,
            'windows': 15,
            'ecgType': 'MECG'
        },
    }

def generate_synthetic_ecg(duration_minutes, fs=500, heart_rate=60, noise_level=0.1):
    """
    Generates a synthetic ECG signal with specified duration and adds baseline wander
    to simulate more realistic ECG noise, with the severity based on a noise level parameter.

    Args:
    - duration_minutes (int): Duration of the ECG signal in minutes.
    - fs (int): Sampling frequency in Hz.
    - heart_rate (int): Heart rate in beats per minute (bpm).
    - noise_level (float): Controls the amplitude and frequency of the baseline wander.

    Returns:
    - np.ndarray: A 2D numpy array representing the ECG signals of two leads with baseline wander.
    """
    duration_seconds = duration_minutes * 60
    t = np.linspace(0, duration_seconds, int(duration_seconds * fs), endpoint=False)

    # Generate base ECG signal
    ecg_clean = nk.ecg_simulate(duration=duration_seconds, sampling_rate=fs, heart_rate=heart_rate)

    # Parameters for baseline wander based on noise_level
    baseline_amplitude = 0.02 + 0.03 * noise_level
    baseline_freq = 0.1 + 2 * noise_level

    # Simulate baseline wander
    baseline_wander = baseline_amplitude * np.sin(2 * np.pi * baseline_freq * t)
    lead1 = ecg_clean + baseline_wander
    lead2 = ecg_clean

    ecg_two_leads = np.vstack([lead1, lead2]).T
    
    return ecg_two_leads

def add_noise_to_ecg(ecg_signal, noise_type="gaussian", noise_level=0.006):
    """
    Adds specified noise to an ECG signal. This is a placeholder for various noise addition strategies.
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, size=ecg_signal.shape)
        return ecg_signal + noise
    return ecg_signal

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Lowpass filter the data.
    
    Args:
    - data (np.ndarray): The ECG signal to be filtered.
    - cutoff (float): The cutoff frequency of the filter.
    - fs (int): The sampling rate of the data.
    - order (int): The order of the filter.
    
    Returns:
    - np.ndarray: The lowpass filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def create_ButterW_bandpass(order, lowcut, highcut, sample_freq):
    sos_params = butter(N=order, Wn=[lowcut, highcut], btype='bandpass', analog=False, output='sos', fs=sample_freq)
    return sos_params

def calculate_sqi_and_compare(ecg_signal, fs):
    """
    Calculates SQI for a given ECG signal using wqrsm for R peak detection and bsqi for SQI calculation.
    
    Args:
    - ecg_signal (np.ndarray): The ECG signal data.
    - fs (int): The sampling frequency of the ECG signal.

    Returns:
    - F1_bsqi (float): The F1 score from the SQI calculation.
    - StartIdxSQIwindows (list): Start indices of SQI windows.
    """
    HRVparams = initialize_HRVparams(fs)
    
    # filtered_ecg_lead1 = lowpass_filter(ecg_signal[:, 0], 100, fs)
    # filtered_ecg_lead2 = lowpass_filter(ecg_signal[:, 1], 100, fs)

    filter = butter(N=5, Wn=[1, 100], btype='bandpass', analog=False, output='sos', fs=fs)
    ecg_signal[:, 0] = sosfiltfilt(sos=filter, x=ecg_signal[:, 0])
    ecg_signal[:, 1] = sosfiltfilt(sos=filter, x=ecg_signal[:, 1])

    # using 'wqrsm', consider using different qrs detector
    # lead1, _ = wqrsm(filtered_ecg_lead1, fs=fs)
    # lead2, _ = wqrsm(filtered_ecg_lead2, fs=fs)

    QRS_detector = Pan_Tompkins_QRS(fs)
    lead1 = QRS_detector.solve(ecg_signal[:, 0])
    lead2 = QRS_detector.solve(ecg_signal[:, 1])
    
    print(lead1)
    print(lead2)
    print()

    F1_bsqi, StartIdxSQIwindows = bsqi.bsqi(lead1, lead2, HRVparams)
    avg_F1_bsqi = np.mean(F1_bsqi)

    return avg_F1_bsqi, StartIdxSQIwindows

def visualize_results(durations, noise_levels, fs=500):
    """
    Visualizes the first 30 seconds of ECG signals for specified durations and noise levels,
    alongside their average F1 SQI scores.

    Args:
    - durations (list): Durations of ECG signals in minutes.
    - noise_levels (list): Noise levels to be added to ECG signals.
    - fs (int): Sampling frequency in Hz.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('First 30 Seconds of ECG Signals with Average F1 SQI Scores', fontsize=16)

    # 10 seconds
    samples_for_10_seconds = 10 * fs

    for i, duration in enumerate(durations):
        for j, noise_level in enumerate(noise_levels):
            ecg_signal = generate_synthetic_ecg(duration, fs=fs, noise_level=noise_level)
            avg_F1_bsqi, _ = calculate_sqi_and_compare(ecg_signal, fs)

            ax = axes[i, j]

            time_axis = np.linspace(0, 10, samples_for_10_seconds)
            signal_length = min(samples_for_10_seconds, ecg_signal.shape[0])
            ax.plot(time_axis[:signal_length], ecg_signal[:signal_length, 0], label="Lead 1")
            ax.plot(time_axis[:signal_length], ecg_signal[:signal_length, 1], label="Lead 2", alpha=0.7)
            
            ax.set_title(f"Duration: {duration} min, Noise: {noise_level}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            ax.text(0.5, -0.25, f"Average F1: {avg_F1_bsqi:.2f}", size=12, ha="center", transform=ax.transAxes)
 
    plt.show()

def main():
    durations = [1, 5]
    noise_levels = [1, 3, 5]

    visualize_results(durations, noise_levels)

if __name__ == "__main__":
    main()
