# config.py
import numpy as np
import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "raw_keypoints")
FILTERED_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "filtered_keypoints")
OPTIMIZED_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "optimized_keypoints")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# --- Input Data Paths (Example - Adjust as needed) ---
# Expects subfolders 'color' and 'depth' within INPUT_DATA_DIR
# Color images named like frame_00000.jpg, frame_00001.jpg, ...
# Depth images named like transformed_depth_image_1.png, transformed_depth_image_2.png, ...
# INPUT_DATA_DIR = "D:\\armz"
INPUT_DATA_DIR = "." # Or specify the absolute path to your data
INPUT_COLOR_SUBDIR = "D:\\FULL_ARMZZZZ_205\\color\\color_frames_16-21-59" # Example
INPUT_DEPTH_SUBDIR = "D:\\FULL_ARMZZZZ_205\\depth\\output_pngs_opencv_16-21-59" # Example
COLOR_FILENAME_FORMAT = "frame_{:05d}.jpg"
DEPTH_FILENAME_FORMAT = "transformed_depth_image_{:d}.png"

# --- Frame Processing ---
# Set START_FRAME and END_FRAME to None to process all found frames
START_FRAME_INDEX = 0 # 0-based index
END_FRAME_INDEX = 325 # Process frames 0 to 324 (inclusive)
FRAME_RATE = 12.0 # Frames per second

# --- Keypoint Detection (MediaPipe & Depth) ---
MP_STATIC_IMAGE_MODE = False
MP_MAX_NUM_HANDS = 2
MP_MIN_DETECTION_CONFIDENCE = 0.15
MP_MIN_TRACKING_CONFIDENCE = 0.15
DEPTH_SEARCH_RADIUS = 3 # Radius for depth sampling neighborhood
# Azure Kinect Intrinsics (Example - Verify for your camera)
FX = 636.6593017578125
FY = 636.251953125
CX = 635.283881879317
CY = 366.8740353496978
# FX = 251.82156
# FY = 251.84393
# CX = 257.42749
# CY = 257.42749
DEPTH_SCALE = 1000.0 # For converting depth map values to meters

# --- Data Dimensions ---
NUM_HANDS = 2
NUM_LANDMARKS_MEDIAPIPE = 21
NUM_LANDMARKS_EXTENDED = 22 # After adding the extra point
NUM_COORDS = 3
Y_OFFSET_FOR_NEW_POINT = -300 # Offset for the 22nd point relative to wrist

# --- Kalman Filtering ---
# Thickness array for initial Z adjustment (optional, based on 1_CalmanFiltering)
# Set to None if not used in detection's read_hand_skeleton_data
THICKNESS_ARRAY = np.array([ 30, 20, 18, 15, 8, 20, 11, 10, 5, 20, 11, 10, 5, 20, 11, 10, 5, 20, 11, 10, 5 ])
APPLY_Z_OPTIMIZATION_PRE_FILTER = False # Use aligned data for Z filtering input?
# Kalman Filter Configuration (Copied from 1_CalmanFiltering.py)
FILTER_CONFIG = {
    'defaults': {
        'X': {  'model_type': 'CV', 'sigma_measurement': 20.5, 'sigma_process_pos': 2.5, 'sigma_initial_pos': 10.0, 'p_value_outlier': 0.005, 'sigma_acceleration': 5.0, 'sigma_initial_vel': 10.0, },
        'Y': { 'model_type': 'CV', 'sigma_measurement': 10.0, 'sigma_process_pos': 1.0, 'sigma_initial_pos': 3.0, 'p_value_outlier': 0.05, 'sigma_acceleration': 5.0, 'sigma_initial_vel': 20.0, },
        'Z': { 'model_type': 'CV', 'sigma_measurement': 3.0, 'sigma_acceleration': 2.0, 'sigma_initial_pos': 5.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.001, 'sigma_process_pos': 0.0, },
    },
    'overrides': {
        ('*', 1, 'X'): { 'model_type': 'CV', 'sigma_measurement': 20.5, 'sigma_process_pos': 2.5, 'sigma_initial_pos': 10.0, 'p_value_outlier': 0.05, 'sigma_acceleration': 15.0, 'sigma_initial_vel': 10.0,},
        ('*', 1, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1, },
        ('*', 1, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01, },
        ('*', 2, 'X'): { 'model_type': 'CV', 'sigma_measurement': 20.5, 'sigma_process_pos': 7.5, 'sigma_initial_pos': 10.0, 'p_value_outlier': 0.05, 'sigma_acceleration': 45.0, 'sigma_initial_vel': 10.0, },
        ('*', 2, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 2, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
        ('*', 3, 'X'): { 'model_type': 'CV', 'sigma_measurement': 0.1, 'sigma_process_pos': 25.5, 'sigma_initial_pos': 50.0, 'p_value_outlier': 0.0, 'sigma_acceleration': 100.01, 'sigma_initial_vel': 20.0, },
        ('*', 3, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 3, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
        ('*', 4, 'X'): { 'model_type': 'CV', 'sigma_measurement': 0.1, 'sigma_process_pos': 25.5, 'sigma_initial_pos': 50.0, 'p_value_outlier': 0.0, 'sigma_acceleration': 100.0, 'sigma_initial_vel': 20.0,},
        ('*', 4, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 4, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
    }
}
APPLY_BUTTERWORTH_POST_FILTER = False # Apply Butterworth to Z after Kalman?
BUTTERWORTH_FS = 8.2
BUTTERWORTH_CUTOFF = 4.0
BUTTERWORTH_ORDER = 7

# --- Skeleton Analysis & Optimization ---
# Define connections for analysis and optimization
# Format: (start_landmark_index, end_landmark_index)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Palm
]
# Target bone lengths for optimization.
# Set to None to calculate from filtered data using skeleton_analyzer.
# Otherwise, provide a dictionary like: {(0, 1): 30, (1, 2): 55, ...}
TARGET_BONE_LENGTHS = None
OPTIMIZATION_METHOD = 'L-BFGS-B'
OPTIMIZATION_MAX_ITER = 30
OPTIMIZATION_FRAME_INDEX = 190 # Example: Optimize only this frame index
# Weights for optimization error terms (adjust for desired behavior)
OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ = 1.5 # Weight for squared distance term
OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS = 5.0  # Weight for absolute distance term (robustness)
OPTIMIZATION_BONE_ERROR_WEIGHT_SQ = 2.0     # Weight for squared bone length difference
OPTIMIZATION_BONE_ERROR_WEIGHT_ABS = 12.0   # Weight for absolute bone length difference (robustness)

# --- Visualization ---
VISUALIZE_FILTERING_OUTPUT = True
VISUALIZATION_PAUSE_DURATION = 0.001 # Seconds between frames, 0 for no pause
VISUALIZATION_SAVE_IMAGES = True
VISUALIZATION_DPI = 100
VISUALIZATION_ELEV = 20.0
VISUALIZATION_AZIM = -75.0
VISUALIZATION_XLIM = (50, 850)
VISUALIZATION_YLIM = (-300, 450)
VISUALIZATION_ZLIM = (-700, -50) # Adjusted based on example

# --- Helper Function ---
def create_dirs():
    """Creates necessary output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(FILTERED_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(OPTIMIZED_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)