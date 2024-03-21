import numpy as np

def CreateWindowRRintervals(tNN, NN, HRVparams, option='normal'):
    """
    Create windows for RR interval analysis.

    Parameters:
    tNN : numpy.array
        Time of the RR interval in seconds.
    NN : numpy.array
        NN interval data in seconds (not directly used in this function).
    HRVparams : dict or object
        Settings for HRV analysis.
    option : str
        Analysis context ('normal', 'af', 'sqi', 'mse', 'dfa').

    Returns:
    windowRRintervals : numpy.array
        Starting time of each window to be analyzed.
    """
    # Set defaults based on the option
    if option in ['af', 'mse', 'dfa', 'sqi', 'HRT']:
        increment = HRVparams[option]['increment']
        windowlength = HRVparams[option]['windowlength']
        if option in ['mse', 'dfa', 'HRT']:  # Adjust for hours if necessary
            increment *= 3600
            windowlength *= 3600
    else:  # Default case, could be customized further
        increment = HRVparams['increment']  # Assuming default increment exists
        windowlength = HRVparams['windowlength']  # Assuming default windowlength exists

    if windowlength > tNN[-1]:  # If the window length is longer than the signal, return empty
        return np.array([0])

    nx = np.floor(tNN[-1]).astype(int)  # Length of sequence
    overlap = windowlength - increment  # Number of overlapping elements
    Nwinds = int((nx - overlap) / (windowlength - overlap))  # Number of windows

    # Starting index of each window
    windowRRintervals = np.arange(0, Nwinds) * (windowlength - overlap)

    return windowRRintervals
