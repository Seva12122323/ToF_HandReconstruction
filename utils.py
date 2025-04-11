# utils.py
import numpy as np
from scipy.signal import butter, lfilter
import logging

# --- Butterworth Filter Functions (from 1_CalmanFiltering.py) ---
def butter_lowpass(cutoff, fs, order=5):
    """Designs a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    # Clip normalized cutoff frequency to avoid errors with butter function
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 1.0 - 1e-6)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth low-pass filter to data."""
    # Ensure data is 1D and has enough points
    if data.ndim != 1 or data.shape[0] <= order * 3:
        # logging.debug(f"Skipping Butterworth: Data dim {data.ndim} or length {data.shape[0]} too short.")
        return data # Return original data if not suitable
    try:
        b, a = butter_lowpass(cutoff, fs, order=order)
        # Use filtfilt for zero-phase filtering if causality is not critical
        # y = filtfilt(b, a, data)
        # Use lfilter for causal filtering (as in original script)
        y = lfilter(b, a, data)
        # Handle potential NaNs introduced by filtering edges
        y = np.nan_to_num(y, nan=data[~np.isnan(data)].mean() if not np.isnan(data).all() else 0.0)
        return y
    except ValueError as e:
        logging.error(f"Error during Butterworth filtering: {e}. Returning original data.", exc_info=True)
        return data # Return original data on error

# --- Z-Optimization Function (from 1_CalmanFiltering.py) ---
# Note: This function's logic might need careful review and adaptation.
# It depends on 'aligned' data which isn't explicitly handled in the main
# detection flow currently.
def opt_choose_dep(z2, z1, z2_aligned, z1_aligned, m):
    """ Helper function potentially used in Z-optimization pre-filtering."""
    # This logic seems designed to choose between raw Z and aligned Z based on difference
    # Thresholds (50*m, 20*m) seem arbitrary without context.
    # If the difference between consecutive raw Z is large, it uses an
    # estimate based on the previous raw Z and the change in aligned Z.
    # Otherwise, it keeps the current raw Z. Needs careful testing if used.
    if (z2 - z1) > 50 * m or (z1 - z2) > 20 * m:
        return (z1 + z2_aligned - z1_aligned)
    else:
        return z2

# --- Add other common utilities if needed ---