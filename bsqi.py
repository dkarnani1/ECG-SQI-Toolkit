import numpy as np
from run_sqi import QRSComparison
from CreateWindowRRIntervals import CreateWindowRRintervals

def bsqi(ann1, ann2, HRVparams=None):
    """
    Computes the Signal Quality Index (SQI) for ECG signals by comparing two sets of QRS detections.

    Parameters:
    ann1 : array_like
        First set of QRS detection annotations, typically an array of integers representing sample indices.
    ann2 : array_like
        Second set of QRS detection annotations, for comparison with the first set.
    HRVparams : dict, optional
        Configuration parameters for SQI calculation, including 'windowlength', 'TimeThreshold', 'margin', and 'Fs' (sampling frequency).

    Returns:
    F1 : numpy.array
        F1 score for the SQI of each analysis window. A higher score indicates better signal quality.
    StartIdxSQIwindows : numpy.array
        Starting indices of the analysis windows used for computing SQI.
    """
    
    if HRVparams is None:
        HRVparams = initialize_HRVparams()

    # Convert annotations to time in seconds
    ann1 = np.array(ann1) / HRVparams['Fs']
    ann2 = np.array(ann2) / HRVparams['Fs']

    # Compute the time range for SQI analysis
    endtime = max(ann1[-1], ann2[-1])
    time = np.arange(1/HRVparams['Fs'], endtime + 1/HRVparams['Fs'], 1/HRVparams['Fs'])

    # Generate analysis windows
    StartIdxSQIwindows = CreateWindowRRintervals(time, [], HRVparams, 'sqi')

    # Initialize F1 score array
    F1 = np.full(len(StartIdxSQIwindows), np.nan)

    # Compute F1 scores for each window
    for seg in range(len(StartIdxSQIwindows)):
        if not np.isnan(StartIdxSQIwindows[seg]):
            # Find annotations within the current window
            idx_ann1_in_win = np.where(
                (ann1 >= StartIdxSQIwindows[seg]) & (ann1 < StartIdxSQIwindows[seg] + HRVparams['sqi']['windowlength'])
            )[0]

            # Calculate the F1 score for the current window
            F1_score, _, _, _ = run_sqi(ann1[idx_ann1_in_win] - StartIdxSQIwindows[seg],
                                        ann2 - StartIdxSQIwindows[seg],
                                        HRVparams['sqi']['TimeThreshold'],
                                        HRVparams['sqi']['margin'],
                                        HRVparams['sqi']['windowlength'],
                                        HRVparams['Fs'])
            F1[seg] = F1_score

    return F1, StartIdxSQIwindows

def run_sqi(a1, a2, threshold, margin, windowlength, fs):
    """
    Placeholder for a function to compute the SQI based on comparison of QRS detections.
    """
    qrs_comp = QRSComparison(a1, a2, thres=threshold, margin=margin, windowlen=windowlength, fs=fs)
    return qrs_comp.run_sqi()

def initialize_HRVparams(Fs=250):
    """
    Initializes default HRV parameters for SQI calculation.
    """
    return {
        'sqi': {
            'windowlength': 10,
            'TimeThreshold': 0.15,
            'margin': 1,
            'increment': 5
        },
        'Fs': Fs,
        'PeakDetect': {
            'REF_PERIOD': 0.250,
            'THRES': 0.6,
            'debug': True,
            'windows': 15,
            'ecgType': 'MECG'
        }
    }

def bsqi_example():
    """
    An example function demonstrating the use of the bsqi function.
    """
    fs = 1000  # Sample rate in Hz
    duration = 60  # Duration in seconds
    
    # Simulate QRS peaks
    ann1 = np.arange(1, duration, 1) * fs
    ann2 = (np.arange(1, duration, 1) + 0.1) * fs

    # Initialize HRV parameters
    HRVparams = initialize_HRVparams(fs)

    # Call bsqi function and print results
    F1, StartIdxSQIwindows = bsqi(ann1, ann2, HRVparams)
    print("F1 scores for SQI:", F1)
    print("StartIdxSQIwindows:", StartIdxSQIwindows)

if __name__ == '__main__':
    # Run the example
    bsqi_example()
