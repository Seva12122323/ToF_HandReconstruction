import numpy as np
import os
import glob
import re 
import config

def find_frame_files(start_index, end_index):
    """Finds corresponding color and depth image files within the specified range."""
    color_dir = os.path.join(config.INPUT_DATA_DIR, config.INPUT_COLOR_SUBDIR)
    depth_dir = os.path.join(config.INPUT_DATA_DIR, config.INPUT_DEPTH_SUBDIR)

    if not os.path.isdir(color_dir) or not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Input color ('{color_dir}') or depth ('{depth_dir}') directory not found.")

    frame_files = []


    glob_format_pattern = re.sub(r"\{.*?\}", "*", config.COLOR_FILENAME_FORMAT)
    color_search_pattern = os.path.join(color_dir, glob_format_pattern)

  
    all_color_files = sorted(glob.glob(color_search_pattern))


    if not all_color_files:
         raise FileNotFoundError(f"No color files found in '{color_dir}' matching pattern '{color_search_pattern}'.")


    try:

        num_extract_regex = re.compile(re.sub(r"\{.*?\}", r"(\d+)", config.COLOR_FILENAME_FORMAT))
        frame_numbers = []
        for f in all_color_files:
            match = num_extract_regex.search(os.path.basename(f))
            if match:
                frame_numbers.append(int(match.group(1))) # Group 1 captures the (\d+) part
            else:
                 print(f"Warning: Could not extract frame number from {os.path.basename(f)} using regex.")

        if not frame_numbers:
             raise ValueError("Could not extract any frame numbers from detected files.")

        max_frame_num = max(frame_numbers)

    except (IndexError, ValueError, re.error) as e:
         print(f"Warning: Could not reliably determine max frame number from filenames using regex ({e}). Using a large default.")
         max_frame_num = 99999 # Fallback

    actual_start_index = config.START_FRAME_INDEX if config.START_FRAME_INDEX is not None else 0
    actual_end_index = config.END_FRAME_INDEX if config.END_FRAME_INDEX is not None else max_frame_num

    print(f"Searching for frames from index {actual_start_index} to {actual_end_index}.")

    for i in range(actual_start_index, actual_end_index + 1):
        frame_num_for_depth = i + 1 
        color_path = os.path.join(color_dir, config.COLOR_FILENAME_FORMAT.format(i)) 
        depth_path = os.path.join(depth_dir, config.DEPTH_FILENAME_FORMAT.format(frame_num_for_depth))

        if os.path.exists(color_path) and os.path.exists(depth_path):
            frame_files.append({'index': i, 'color': color_path, 'depth': depth_path})


    if not frame_files:
        raise FileNotFoundError(f"No corresponding color/depth file pairs found in the specified range/directories.")

    print(f"Found {len(frame_files)} corresponding frame pairs.")
    return frame_files, actual_start_index, actual_end_index # Return actual range used

def save_keypoints_to_file(filepath, frame_keypoints, frame_index, num_landmarks_to_save):
    """
    Saves keypoints for a single frame to a text file.

    Args:
        filepath (str): Path to the output text file.
        frame_keypoints (np.ndarray): Keypoints for the frame, shape (num_hands, num_landmarks, 3).
        frame_index (int): Original index of the frame.
        num_landmarks_to_save (int): Number of landmarks per hand to save.
    """
    try:
        num_hands = frame_keypoints.shape[0]
        with open(filepath, "w") as f:
            f.write(f"# Frame Index: {frame_index}\n") 
            for hand_idx in range(num_hands):
                f.write(f"Hand {hand_idx}\n")
                landmarks_this_hand = min(num_landmarks_to_save, frame_keypoints.shape[1])
                for landmark_idx in range(landmarks_this_hand):
                    coords = frame_keypoints[hand_idx, landmark_idx, :]
                    if np.isnan(coords).any():
                        coord_str = "x=nan, y=nan, z=nan"
                    else:
                        coord_str = f"x={coords[0]:.6f}, y={coords[1]:.6f}, z={coords[2]:.6f}"
                    f.write(f"  Landmark {landmark_idx}: {coord_str}\n")
    except IOError as e:
        print(f"Error writing file {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error saving frame {frame_index} data to {filepath}: {e}")


def load_keypoints_from_file(filepath, expected_hands, expected_landmarks):
    """
    Loads 3D coordinates for a single frame from a text file.
    Handles the format written by save_keypoints_to_file and the original format.

    Args:
        filepath (str): Path to the input text file.
        expected_hands (int): Expected number of hands.
        expected_landmarks (int): Expected number of landmarks per hand.

    Returns:
        np.ndarray | None: NumPy array shape (expected_hands, expected_landmarks, 3)
                         or None if file not found or critical error.
                         Returns array with NaNs for missing/unparsed data.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    frame_data = np.full((expected_hands, expected_landmarks, 3), np.nan, dtype=float)
    landmarks_found_per_hand = {h: 0 for h in range(expected_hands)}
    current_hand_index = -1


    hand_regex = re.compile(r"Hand\s*(\d+)")
    landmark_regex = re.compile(r"Landmark\s*(\d+):\s*x=([-\d.nan]+),\s*y=([-\d.nan]+),\s*z=([-\d.nan]+)") # Allow 'nan'

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue

                hand_match = hand_regex.match(line)
                if hand_match:
                    try:
                        current_hand_index = int(hand_match.group(1))
                        
                        if current_hand_index >= expected_hands and current_hand_index == 1 and expected_hands == 1:
                             current_hand_index = 0 
                        elif current_hand_index >= expected_hands: 
                             if current_hand_index -1 < expected_hands:
                                 current_hand_index = current_hand_index -1 
                             else:
                                 print(f"Warning in {filepath}: Invalid hand index {hand_match.group(1)} on line {line_num}. Expected 0-{expected_hands-1}. Skipping hand.")
                                 current_hand_index = -1
                                 continue
                        if not (0 <= current_hand_index < expected_hands):
                             print(f"Warning in {filepath}: Invalid hand index {current_hand_index} on line {line_num}. Expected 0-{expected_hands-1}. Skipping hand.")
                             current_hand_index = -1
                        continue
                    except ValueError:
                        print(f"Warning in {filepath}: Could not parse hand index from line {line_num}: '{line}'.")
                        current_hand_index = -1
                        continue

                landmark_match = landmark_regex.match(line)
                if landmark_match:
                    if current_hand_index == -1:
                        continue

                    try:
                        lm_idx = int(landmark_match.group(1))
                        x_str = landmark_match.group(2).lower()
                        y_str = landmark_match.group(3).lower()
                        z_str = landmark_match.group(4).lower()

                        x = float(x_str) if x_str != 'nan' else np.nan
                        y = float(y_str) if y_str != 'nan' else np.nan
                        z = float(z_str) if z_str != 'nan' else np.nan

                        if 0 <= lm_idx < expected_landmarks:
                            frame_data[current_hand_index, lm_idx, :] = [x, y, z]
                            landmarks_found_per_hand[current_hand_index] += 1
                            pass

                    except ValueError as e:
                        print(f"Warning in {filepath}: Could not parse landmark index or coords on line {line_num}: '{line}'. Error: {e}. Setting to NaN.")
                        if 0 <= lm_idx < expected_landmarks:
                             frame_data[current_hand_index, lm_idx, :] = [np.nan, np.nan, np.nan]
                    except IndexError:
                        print(f"Warning in {filepath}: Data structure error accessing frame_data for hand {current_hand_index}, landmark {lm_idx}. Skipping landmark.")
                    continue
        return frame_data

    except Exception as e:
        print(f"Error reading or processing file {filepath}: {e}")
        return None


def load_all_keypoints(input_dir, file_pattern, expected_hands, expected_landmarks):
    """
    Loads keypoint data from all matching files in a directory into a single NumPy array.

    Argf hands per frame.
        expected_landmarks (int): Expected numbs:
        input_dir (str): Directory containing the keypoint files.
        file_pattern (str): Glob pattern for the files (e.g., '*_raw_keypoints.txt').
        expected_hands (int): Expected number oer of landmarks per hand.

    Returns:
        tuple: (np.ndarray | None, list[int])
               - NumPy array shape (num_frames, expected_hands, expected_landmarks, 3) or None on error.
               - List of frame indices corresponding to the loaded frames.
    """
    print(f"Loading all keypoints from: {input_dir} using pattern: {file_pattern}")
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return None, []

    search_path = os.path.join(input_dir, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files matching pattern '{search_path}' found.")
        return None, []

    # Sort files by frame index extracted from the filename (assuming format like 'INDEX_*.txt')
    def get_frame_index_from_path(path):
        filename = os.path.basename(path)
        try:
            return int(filename.split('_')[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not extract frame index from filename: {filename}. Skipping.")
            return -1

    sorted_file_paths = sorted([p for p in file_paths if get_frame_index_from_path(p) != -1],
                               key=get_frame_index_from_path)

    if not sorted_file_paths:
        print("Error: No valid file paths found after attempting to extract frame indices.")
        return None, []

    all_frames_data = []
    loaded_frame_indices = []
    files_with_errors = 0

    for file_path in sorted_file_paths:
        frame_index = get_frame_index_from_path(file_path)
        frame_data = load_keypoints_from_file(file_path, expected_hands, expected_landmarks)

        if frame_data is not None:
            if frame_data.shape == (expected_hands, expected_landmarks, 3):
                all_frames_data.append(frame_data)
                loaded_frame_indices.append(frame_index)
            else:
                print(f"Error: Loaded data from {file_path} has incorrect shape {frame_data.shape}. Expected {(expected_hands, expected_landmarks, 3)}. Skipping frame.")
                files_with_errors += 1
        else:
            print(f"Error loading data from {file_path}. Skipping frame.")
            files_with_errors += 1

    if not all_frames_data:
        print("Error: No data could be loaded successfully from any files.")
        return None, []

    try:
        final_data = np.array(all_frames_data, dtype=float)
        expected_shape = (len(loaded_frame_indices), expected_hands, expected_landmarks, 3)
        if final_data.shape != expected_shape:
            print(f"Error: Final assembled data shape {final_data.shape} does not match expected {expected_shape}.")
            return None, []

        print(f"Successfully loaded data for {final_data.shape[0]} frames.")
        if files_with_errors > 0:
            print(f"Encountered errors in {files_with_errors} files.")
        nan_count = np.isnan(final_data).sum()
        if nan_count > 0:
            print(f"Warning: Loaded data contains {nan_count} NaN values.")

        return final_data, loaded_frame_indices

    except ValueError as e:
        print(f"Error converting loaded data to NumPy array: {e}. Check for inconsistencies.")
        return None, []
    except Exception as e:
        print(f"An unexpected error occurred during final data assembly: {e}")
        return None, []