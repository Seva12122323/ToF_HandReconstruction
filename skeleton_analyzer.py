# skeleton_analyzer.py
import numpy as np
import config # Import project configuration
import logging

def calculate_median_bone_lengths(filtered_data, connections):
    """
    Calculates the median length for each specified bone connection across all frames.

    Args:
        filtered_data (np.ndarray): Filtered keypoint data, shape
                                    (num_frames, num_hands, num_landmarks, 3).
                                    Assumes NUM_LANDMARKS >= max index in connections.
        connections (list[tuple]): List of bone connections, e.g., [(0, 1), (1, 2), ...].

    Returns:
        dict: Dictionary mapping connection tuple to its median length (float).
              Returns None if input data is invalid.
    """
    if filtered_data is None or filtered_data.ndim != 4:
        logging.error("Invalid input data for calculating bone lengths.")
        return None

    num_frames, num_hands, num_landmarks, _ = filtered_data.shape
    if num_frames == 0:
        logging.warning("No frames in data to calculate bone lengths.")
        return {}

    median_lengths = {}
    logging.info(f"Calculating median bone lengths for {len(connections)} connections over {num_frames} frames...")

    for start_idx, end_idx in connections:
        if start_idx >= num_landmarks or end_idx >= num_landmarks:
            logging.warning(f"Connection ({start_idx}, {end_idx}) skipped: Index out of bounds for {num_landmarks} landmarks.")
            continue

        all_lengths = []
        # Calculate length for each hand and each frame
        for frame_idx in range(num_frames):
            for hand_idx in range(num_hands):
                p_start = filtered_data[frame_idx, hand_idx, start_idx]
                p_end = filtered_data[frame_idx, hand_idx, end_idx]

                # Calculate distance only if both points are valid (not NaN)
                if not np.isnan(p_start).any() and not np.isnan(p_end).any():
                    dist = np.linalg.norm(p_end - p_start)
                    # Add a check for non-zero length? Sometimes detection might yield coincident points.
                    if dist > 1e-6: # Avoid zero lengths if possible
                         all_lengths.append(dist)

        if all_lengths:
            median_len = np.median(all_lengths)
            median_lengths[(start_idx, end_idx)] = median_len
            # logging.debug(f"Median length for connection {start_idx}-{end_idx}: {median_len:.2f} (from {len(all_lengths)} samples)")
        else:
            logging.warning(f"No valid length samples found for connection {start_idx}-{end_idx}. Assigning NaN.")
            median_lengths[(start_idx, end_idx)] = np.nan

    logging.info("Median bone length calculation complete.")
    return median_lengths