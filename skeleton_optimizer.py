# skeleton_optimizer.py
import numpy as np
from scipy.optimize import minimize
import config # Import project configuration
import logging
import time

def landmark_error(landmarks_opt, landmarks_detected):
    """Calculates the error between optimized and detected landmarks."""
    # Ensure landmarks_detected is also (N, 3)
    if landmarks_opt.shape != landmarks_detected.shape:
         raise ValueError("Shape mismatch between optimized and detected landmarks.")

    diff = landmarks_opt - landmarks_detected
    # Use robust error (sum of weighted squared and absolute differences)
    # Ignore NaNs in detected landmarks
    valid_mask = ~np.isnan(landmarks_detected).any(axis=1)
    if not np.any(valid_mask): return 0.0 # No valid landmarks to compare

    diff_valid = diff[valid_mask]
    error_sq = np.sum(config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ * (diff_valid**2))
    error_abs = np.sum(config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS * np.sqrt(np.sum(diff_valid**2, axis=1))) # L1 norm on vector diff

    return error_sq + error_abs


def bone_length_constraint_error(landmarks_opt, target_bone_lengths):
    """Calculates the error based on deviation from target bone lengths."""
    constraint_error = 0.0
    num_landmarks = landmarks_opt.shape[0]

    for (start, end), target_length in target_bone_lengths.items():
        if start >= num_landmarks or end >= num_landmarks:
             logging.warning(f"Skipping bone constraint ({start},{end}): index out of bounds.")
             continue
        if np.isnan(target_length):
             logging.warning(f"Skipping bone constraint ({start},{end}): target length is NaN.")
             continue

        p_start = landmarks_opt[start]
        p_end = landmarks_opt[end]

        # Check if landmark positions themselves are valid before calculating distance
        if np.isnan(p_start).any() or np.isnan(p_end).any():
            # Penalize if optimization leads to NaN positions? Or just skip?
            # Skipping seems safer. The landmark_error should handle detected NaNs.
            continue

        dist = np.linalg.norm(p_start - p_end)
        diff = dist - target_length

        error_sq = config.OPTIMIZATION_BONE_ERROR_WEIGHT_SQ * (diff**2)
        error_abs = config.OPTIMIZATION_BONE_ERROR_WEIGHT_ABS * np.abs(diff) # L1 norm on scalar diff

        constraint_error += error_sq + error_abs

    return constraint_error

def total_error(x_flat, target_bone_lengths, detected_landmarks, num_landmarks):
    """Total error function for the optimizer."""
    landmarks_opt = x_flat.reshape((num_landmarks, 3))

    error_lm = landmark_error(landmarks_opt, detected_landmarks)
    error_bone = bone_length_constraint_error(landmarks_opt, target_bone_lengths)

    return error_lm + error_bone

def optimize_skeleton(initial_landmarks, target_bone_lengths, detected_landmarks, k_iter):
    """
    Optimizes landmark positions to satisfy bone length constraints while
    staying close to detected positions.

    Args:
        initial_landmarks (np.ndarray): Starting guess for optimization,
                                        shape (num_landmarks, 3). Usually the
                                        filtered landmarks for the frame.
        target_bone_lengths (dict): Dictionary mapping (start, end) tuple to target length.
        detected_landmarks (np.ndarray): The original detected/filtered landmarks
                                         for this frame, shape (num_landmarks, 3). Used
                                         in the error function.

    Returns:
        np.ndarray: Optimized landmark positions, shape (num_landmarks, 3).
                    Returns initial_landmarks if optimization fails.
    """
    if initial_landmarks is None or detected_landmarks is None or target_bone_lengths is None:
        logging.error("Missing input for optimization.")
        return initial_landmarks # Return initial guess on error

    num_landmarks = initial_landmarks.shape[0]
    if num_landmarks == 0:
        logging.warning("No landmarks provided for optimization.")
        return initial_landmarks

    # Flatten initial guess for the optimizer
    initial_values = initial_landmarks.flatten()

    # Handle NaNs in initial guess or detected landmarks:
    # Option 1: Replace NaNs with a default value (e.g., 0 or mean) before optimization.
    # Option 2: Modify error functions to ignore NaNs (already partially done).
    # Option 3: Skip optimization if too many NaNs.
    # Let's proceed with Option 2, assuming error functions handle NaNs.
    if np.isnan(initial_values).any():
         logging.warning("NaNs found in initial guess for optimization. Proceeding, but results may be affected.")
         # Optionally replace NaNs here: initial_values = np.nan_to_num(initial_values)

    logging.debug(f"Starting optimization for {num_landmarks} landmarks...")
    optimization_start_time = time.time()

    try:
        result = minimize(
            total_error,
            initial_values,
            args=(target_bone_lengths, detected_landmarks, num_landmarks),
            method=config.OPTIMIZATION_METHOD,
            options={'disp': False, 'maxiter': int((config.OPTIMIZATION_MAX_ITER)*k_iter)} # disp=False for less console output
        )
        optimization_end_time = time.time()

        if result.success:
            optimized_landmarks = result.x.reshape((num_landmarks, 3))
            logging.debug(f"Optimization successful in {optimization_end_time - optimization_start_time:.3f}s. Final error: {result.fun:.4f}")
            return optimized_landmarks
        else:
            logging.warning(f"Optimization failed: {result.message}")
            return initial_landmarks # Return initial guess if optimization fails

    except Exception as e:
        logging.error(f"Error during optimization: {e}", exc_info=True)
        return initial_landmarks # Return initial guess on exception