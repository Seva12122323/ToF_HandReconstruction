# main.py
import os
import time
import numpy as np
import logging

# Import project modules
import config
import data_io
from keypoint_detector import KeypointDetector
from kalman_filter_module import FilteringPipeline
from skeleton_analyzer import calculate_median_bone_lengths
from skeleton_optimizer import optimize_skeleton
import visualization # Import the whole module for clarity

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()]) # Output logs to console

def run_pipeline():
    """Executes the full hand pose estimation and refinement pipeline."""
    start_time_total = time.time()
    logging.info("Starting Hand Pose Pipeline...")

    # --- 0. Setup ---
    config.create_dirs() # Create output directories if they don't exist
    logging.info(f"Using output directory: {config.OUTPUT_DIR}")

    # --- 1. Find Input Frames ---
    try:
        frame_files, start_idx, end_idx = data_io.find_frame_files(
            config.START_FRAME_INDEX, config.END_FRAME_INDEX
        )
        num_frames_to_process = len(frame_files)
        if num_frames_to_process == 0:
             logging.error("No frames found to process. Exiting.")
             return
    except FileNotFoundError as e:
        logging.error(f"Error finding input files: {e}. Exiting.")
        return

    # --- 2. Keypoint Detection ---
    logging.info("--- Stage 1: Keypoint Detection ---")
    detector = KeypointDetector()
    detection_errors = 0
    start_time_detection = time.time()

    for frame_info in frame_files:
        frame_idx = frame_info['index']
        # logging.info(f"Detecting frame {frame_idx}...")
        try:
            # Returns list of arrays, one per hand (padded with NaNs if needed)
            detected_hands_3d = detector.process_frame(
                frame_info['color'], frame_info['depth']
            )

            # Combine into a single array for saving (hands, landmarks, coords)
            frame_keypoints_raw = np.stack(detected_hands_3d, axis=0)

            # Save raw keypoints for this frame
            raw_filename = f"{frame_idx}_raw_keypoints.txt"
            raw_filepath = os.path.join(config.RAW_KEYPOINTS_DIR, raw_filename)
            data_io.save_keypoints_to_file(raw_filepath, frame_keypoints_raw, frame_idx, config.NUM_LANDMARKS_MEDIAPIPE)

        except Exception as e:
            logging.error(f"Error detecting keypoints for frame {frame_idx}: {e}", exc_info=True)
            detection_errors += 1
            # Save empty/NaN file to maintain sequence?
            error_data = np.full((config.NUM_HANDS, config.NUM_LANDMARKS_MEDIAPIPE, 3), np.nan)
            raw_filename = f"{frame_idx}_raw_keypoints.txt"
            raw_filepath = os.path.join(config.RAW_KEYPOINTS_DIR, raw_filename)
            data_io.save_keypoints_to_file(raw_filepath, error_data, frame_idx, config.NUM_LANDMARKS_MEDIAPIPE)


    detector.close()
    end_time_detection = time.time()
    logging.info(f"Keypoint detection finished in {end_time_detection - start_time_detection:.2f}s. Errors: {detection_errors}")
    end_time_total = time.time()
    logging.info(f"Detection finished in {end_time_total - start_time_total:.2f} seconds.")
    # --- 3. Load All Raw Keypoints ---
    logging.info("Loading all detected raw keypoints...")
    all_raw_data, loaded_indices_raw = data_io.load_all_keypoints(
        config.RAW_KEYPOINTS_DIR,
        '*_raw_keypoints.txt',
        config.NUM_HANDS,
        config.NUM_LANDMARKS_MEDIAPIPE
    )
    if all_raw_data is None:
        logging.error("Failed to load raw keypoints for filtering. Exiting.")
        return
    if len(loaded_indices_raw) != num_frames_to_process:
         logging.warning(f"Number of loaded raw files ({len(loaded_indices_raw)}) differs from initially found frames ({num_frames_to_process}).")


    # --- 4. Kalman Filtering ---
    logging.info("--- Stage 2: Kalman Filtering ---")
    filter_pipeline = FilteringPipeline(config.FILTER_CONFIG, config.FRAME_RATE)
    try:
        # Apply filtering (includes initialization, filtering, adding 22nd point)
        all_filtered_data = filter_pipeline.apply_filter(all_raw_data) # Shape includes 22 landmarks
    except Exception as e:
        logging.error(f"Error during Kalman filtering: {e}. Exiting.", exc_info=True)
        return
    end_time_total = time.time()
    logging.info(f"Kalman finished in {end_time_total - start_time_total:.2f} seconds.")
    # --- 5. Save Filtered Keypoints ---
    logging.info("Saving filtered keypoints...")
    save_errors_filtered = 0
    for i, frame_idx in enumerate(loaded_indices_raw): # Use indices from loaded raw data
        filtered_filename = f"{frame_idx}_filtered_keypoints.txt"
        filtered_filepath = os.path.join(config.FILTERED_KEYPOINTS_DIR, filtered_filename)
        try:
            # Save data with the extended number of landmarks
            data_io.save_keypoints_to_file(filtered_filepath, all_filtered_data[i], frame_idx, config.NUM_LANDMARKS_EXTENDED)
        except Exception:
             save_errors_filtered += 1 # Error already logged by save function

    if save_errors_filtered > 0:
         logging.warning(f"Encountered {save_errors_filtered} errors while saving filtered data.")

    # --- 6. Optional: Visualize Filtered Output ---
    if config.VISUALIZE_FILTERING_OUTPUT:
        logging.info("Visualizing filtered keypoints...")
        vis_start_time = time.time()
        for i, frame_idx in enumerate(loaded_indices_raw):
            if i % 50 == 0: logging.info(f"  Visualizing frame {i+1}/{len(loaded_indices_raw)} (Index {frame_idx})")
            visualization.visualize_frame(all_filtered_data[i], frame_idx, title_suffix="Filtered")
        vis_end_time = time.time()
        logging.info(f"Visualization finished in {vis_end_time - vis_start_time:.2f}s.")

    end_time_total = time.time()
    logging.info(f"Visualization finished in {end_time_total - start_time_total:.2f} seconds.")
    # --- 7. Skeleton Analysis (Calculate Bone Lengths) ---
    logging.info("--- Stage 3: Skeleton Analysis ---")
    target_lengths = config.TARGET_BONE_LENGTHS
    if target_lengths is None:
        logging.info("Calculating median bone lengths from filtered data...")
        # Use the filtered data *before* the 22nd point was added if connections only use 0-20
        data_for_analysis = all_filtered_data[:, :, :config.NUM_LANDMARKS_MEDIAPIPE, :]
        calculated_lengths = calculate_median_bone_lengths(data_for_analysis, config.SKELETON_CONNECTIONS)
        if calculated_lengths:
            logging.info("Calculated Median Bone Lengths:")
            for conn, length in calculated_lengths.items():
                logging.info(f"  {conn}: {length:.2f}")
            target_lengths = calculated_lengths
        else:
            logging.error("Failed to calculate bone lengths. Cannot proceed with optimization if lengths were not predefined.")
            target_lengths = {} # Assign empty dict to avoid error later if optimization is attempted
    else:
        logging.info("Using predefined target bone lengths.")

    # --- 8. Skeleton Optimization ---
    logging.info("--- Stage 4: Skeleton Optimization ---")
    if not target_lengths:
        logging.warning("Skipping optimization: Target bone lengths are not available.")
    else:
        # Example: Optimize only a specific frame or a range
        frame_to_optimize_idx = config.OPTIMIZATION_FRAME_INDEX # Get index from config
        optimize_errors = 0
        opt_start_time = time.time()

        # Find the array index corresponding to the desired frame index
        try:
            # array_idx = loaded_indices_raw.index(frame_to_optimize_idx)
            for array_idx in range(len(loaded_indices_raw)):
                logging.info(f"Optimizing frame index {frame_to_optimize_idx} (Array index {array_idx})...")

                # Get the filtered data for this specific frame (initial guess)
                # Use data with 22 points if optimization needs it, otherwise slice
                initial_guess = all_filtered_data[array_idx] # Shape (hands, 22, 3)
                detected_frame = all_filtered_data[array_idx] # Use filtered as 'detected' for error term

                optimized_frame_data = np.full_like(initial_guess, np.nan)
                k_iter=1
                for hand_idx in range(config.NUM_HANDS):
                    # Slice if optimizer only works on 21 landmarks
                    num_landmarks_for_opt = initial_guess.shape[1] # Use all landmarks (22)
                    # num_landmarks_for_opt = config.NUM_LANDMARKS_MEDIAPIPE # Use only 21
                    if array_idx>10:
                        initial_hand = initial_guess[hand_idx-1, :num_landmarks_for_opt, :]
                        k_iter=0.5
                    else:
                        initial_hand = initial_guess[hand_idx, :num_landmarks_for_opt, :]
                    detected_hand = detected_frame[hand_idx, :num_landmarks_for_opt, :]

                    # Check for too many NaNs before optimizing
                    if np.isnan(initial_hand).sum() > initial_hand.size * 0.5:
                        logging.warning(f"Skipping optimization for hand {hand_idx} in frame {frame_to_optimize_idx}: Too many NaNs in initial guess.")
                        optimized_frame_data[hand_idx, :num_landmarks_for_opt, :] = initial_hand # Keep initial
                        continue

                    optimized_hand = optimize_skeleton(
                        initial_hand,
                        target_lengths,
                        detected_hand, # Pass the corresponding filtered data as the reference,
                        k_iter
                    )
                    optimized_frame_data[hand_idx, :num_landmarks_for_opt, :] = optimized_hand
                    # If 22nd point wasn't optimized, copy it over? Or recalculate?
                    if num_landmarks_for_opt < config.NUM_LANDMARKS_EXTENDED:
                        optimized_frame_data[hand_idx, num_landmarks_for_opt:, :] = initial_guess[hand_idx, num_landmarks_for_opt:, :]


                # Save optimized keypoints
                opt_filename = f"{frame_to_optimize_idx}_optimized_keypoints.txt"
                opt_filepath = os.path.join(config.OPTIMIZED_KEYPOINTS_DIR, opt_filename)
                data_io.save_keypoints_to_file(opt_filepath, optimized_frame_data, frame_to_optimize_idx, config.NUM_LANDMARKS_EXTENDED)

                # Visualize optimized frame
                visualization.visualize_frame(optimized_frame_data, frame_to_optimize_idx, title_suffix="Optimized")

        except ValueError:
            logging.error(f"Frame index {frame_to_optimize_idx} specified for optimization not found in loaded frames.")
            optimize_errors += 1
        except Exception as e:
            logging.error(f"Error during optimization for frame {frame_to_optimize_idx}: {e}", exc_info=True)
            optimize_errors += 1

        opt_end_time = time.time()
        logging.info(f"Optimization finished in {opt_end_time - opt_start_time:.2f}s. Errors: {optimize_errors}")


    # --- End of Pipeline ---
    end_time_total = time.time()
    logging.info(f"Pipeline finished in {end_time_total - start_time_total:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()