import numpy as np
from scipy.signal import butter, lfilter
import logging

# Butterworth Filter 
def butter_lowpass(cutoff, fs, order=5):
    """Designs a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 1.0 - 1e-6)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth low-pass filter to data."""
    if data.ndim != 1 or data.shape[0] <= order * 3:
        return data 
    try:
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        y = np.nan_to_num(y, nan=data[~np.isnan(data)].mean() if not np.isnan(data).all() else 0.0)
        return y
    except ValueError as e:
        logging.error(f"Error during Butterworth filtering: {e}. Returning original data.", exc_info=True)
        return data 

def opt_choose_dep(z2, z1, z2_aligned, z1_aligned, m):
    """ Helper function potentially used in Z-optimization pre-filtering."""
    if (z2 - z1) > 50 * m or (z1 - z2) > 20 * m:
        return (z1 + z2_aligned - z1_aligned)
    else:
        return z2

