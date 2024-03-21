import wfdb
import numpy as np
from wqrsm import wqrsm
from run_sqi import QRSComparison
import bsqi
import pandas as pd
from jqrs import jqrs

# Function to initialize HRV parameters
def initialize_HRVparams(fs):
    return {
        'sqi': {
            'windowlength': 180,
            'TimeThreshold': 0.1,
            'margin': 0.05,
            'increment': 100
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

# Function to read ECG data from a .dat file
def read_ecg_dat_file(file_path):
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal
    return signal, record.fs

# Load ECG data
file_path = "JS07300"
ecg_data, fs = read_ecg_dat_file('test ECGs\\101')

# Initialize HRV parameters
HRVparams = initialize_HRVparams(fs)

# Run QRS detection using wqrsm
qrs_wqrsm1, _ = wqrsm(ecg_data[:, 0], fs=fs)
qrs_wqrsm2, _ = wqrsm(ecg_data[:, 1], fs=fs)

# Run QRS detection using jqrs
# qrs_wqrsm1, _, _ = jqrs(ecg_data[:, 0], HRVparams)
# qrs_wqrsm2, _, _ = jqrs(ecg_data[:, 1], HRVparams)

# Compute SQI using bsqi function
F1_bsqi, StartIdxSQIwindows = bsqi.bsqi(qrs_wqrsm1, qrs_wqrsm2, HRVparams)

# Print bsqi results
print("F1 score (bsqi):", F1_bsqi)
print("Start Index SQI windows:", StartIdxSQIwindows)
