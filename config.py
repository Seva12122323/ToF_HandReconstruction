import numpy as np
import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "raw_keypoints")
FILTERED_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "filtered_keypoints")
OPTIMIZED_KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "optimized_keypoints")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# --- Input Data Paths ---

INPUT_DATA_DIR = "." # Or specify the absolute path to your data
INPUT_COLOR_SUBDIR = "D:\\FULL_ARMZZZZ_205\\color\\color_frames_16-21-59" 
INPUT_DEPTH_SUBDIR = "D:\\FULL_ARMZZZZ_205\\depth\\output_pngs_opencv_16-21-59" # Example
COLOR_FILENAME_FORMAT = "frame_{:05d}.jpg"
DEPTH_FILENAME_FORMAT = "transformed_depth_image_{:d}.png"

# --- Frame Processing ---
START_FRAME_INDEX = 0 
END_FRAME_INDEX = 324 
FRAME_RATE = 30.0 

# --- Keypoint Detection (MediaPipe & Depth) ---
MP_STATIC_IMAGE_MODE = True
MP_MAX_NUM_HANDS = 2
MP_MIN_DETECTION_CONFIDENCE = 0.15
MP_MIN_TRACKING_CONFIDENCE = 0.15
DEPTH_SEARCH_RADIUS = 3 # Radius for depth sampling neighborhood
# Camera params
FX = 636.6593017578125
FY = 636.251953125
CX = 635.283881879317
CY = 366.8740353496978
DEPTH_SCALE = 1000.0 # For converting depth map values to meters

# --- Data Dimensions ---
NUM_HANDS = 2
NUM_LANDMARKS_MEDIAPIPE = 21
NUM_LANDMARKS_EXTENDED = 22 # After adding the extra point if need
NUM_COORDS = 3
Y_OFFSET_FOR_NEW_POINT = -300 # Offset for the 22nd point relative to wrist if we just add without calculations

# --- Kalman Filtering ---
THICKNESS_ARRAY = np.array([ 30, 20, 18, 15, 8, 20, 11, 10, 5, 20, 11, 10, 5, 20, 11, 10, 5, 20, 11, 10, 5 ]) # Thickness ups to the joint
APPLY_Z_OPTIMIZATION_PRE_FILTER = False # Use aligned data for Z filtering acording to trashold?
# Kalman Filter Configuration CP - constant position CV - constant velocity
FILTER_CONFIG = {
    'defaults': {
        'X': {  'model_type': 'CV', 'sigma_measurement': 20.5, 'sigma_process_pos': 2.5, 'sigma_initial_pos': 10.0, 'p_value_outlier': 0.005, 'sigma_acceleration': 5.0, 'sigma_initial_vel': 10.0, },
        'Y': { 'model_type': 'CV', 'sigma_measurement': 20.0, 'sigma_process_pos': 1.5, 'sigma_initial_pos': 11.0, 'p_value_outlier': 0.01, 'sigma_acceleration': 5.0, 'sigma_initial_vel': 10.0, },
        'Z': { 'model_type': 'CP', 'sigma_measurement': 0.4, 'sigma_acceleration': .5, 'sigma_initial_pos': 1.0, 'sigma_initial_vel': 5.0, 'p_value_outlier': 0.005, 'sigma_process_pos': 0.0, },
    },
    'overrides': {
        ('*', 1, 'X'): { 'model_type': 'CV', 'sigma_measurement': 20.5, 'sigma_process_pos': 5.5, 'sigma_initial_pos': 40.0, 'p_value_outlier': 0.05, 'sigma_acceleration': 55.0, 'sigma_initial_vel': 10.0,},
        ('*', 1, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1, },
        ('*', 1, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01, },
        ('*', 2, 'X'): { 'model_type': 'CV', 'sigma_measurement': 0.5, 'sigma_process_pos': 20.5, 'sigma_initial_pos': 50.0, 'p_value_outlier': 0.0, 'sigma_acceleration': 75.0, 'sigma_initial_vel': 10.0, },
        ('*', 2, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 2, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
        ('*', 3, 'X'): { 'model_type': 'CV', 'sigma_measurement': 0.1, 'sigma_process_pos': 35.5, 'sigma_initial_pos': 50.0, 'p_value_outlier': 0.00, 'sigma_acceleration': 70.01, 'sigma_initial_vel': 20.0, },
        ('*', 3, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 3, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
        ('*', 4, 'X'): { 'model_type': 'CV', 'sigma_measurement': 0.1, 'sigma_process_pos': 35.5, 'sigma_initial_pos': 50.0, 'p_value_outlier': 0.0, 'sigma_acceleration': 100.0, 'sigma_initial_vel': 20.0,},
        ('*', 4, 'Y'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 4, 'Z'): { 'model_type': 'CV', 'sigma_measurement': 30.0, 'sigma_acceleration': 30.0, 'sigma_initial_pos': 90.0, 'sigma_initial_vel': 20.0, 'p_value_outlier': 0.01 },
        ('*', 5, 'Z'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
        ('*', 9, 'Z'): { 'model_type': 'CP', 'sigma_measurement': 0.2, 'sigma_process_pos': 40.0, 'sigma_initial_pos': 0.6, 'p_value_outlier': 0.1 },
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


OPTIMIZATION_METHOD = 'SLSQP' # for fix length
# OPTIMIZATION_METHOD = 'L-BFGS-B' # for not fix length
OPTIMIZATION_MAX_ITER = 1000
OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ = 1.5 # Weight for squared distance term
OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS = 5.0 # Weight for absolute distance term (robustness)
OPTIMIZATION_BONE_ERROR_WEIGHT_SQ = 2.0    # Weight for squared bone length difference
OPTIMIZATION_BONE_ERROR_WEIGHT_ABS = 12.0   # Weight for absolute bone length difference (robustness)
OPTIMIZATION_BONE_LENGTH_ERROR_WEIGHT=  3 

OPTIMIZATION_MAX_ITER = 200 
TARGET_BONE_LENGTHS=None
# TARGET_BONE_LENGTHS= {(0, 1): 36, (1, 2): 32, (2, 3): 28, (3, 4): 23, (0, 5): 71, (5, 6): 37, (6, 7): 23, (7, 8): 18, (5, 9): 21, (9, 10): 39, (10, 11): 25, (11, 12): 18, (9, 13): 19, (13, 14): 36, (14, 15): 23, (15, 16): 17, (13, 17): 19, (17, 18): 28, (18, 19): 18, (19, 20): 14, (0, 17): 64}
OPTIMIZATION_FRAME_INDEX = 190 # Outdated param

# --- Visualization ---
VISUALIZE_FILTERING_OUTPUT = False
VISUALIZATION_PAUSE_DURATION = 0.001 
VISUALIZATION_SAVE_IMAGES = True
VISUALIZATION_DPI = 100
VISUALIZATION_ELEV = 20.0
VISUALIZATION_AZIM = -75.0
VISUALIZATION_XLIM = (50, 300)
VISUALIZATION_YLIM = (-50, 450)
VISUALIZATION_ZLIM = (-700, -250) 
VISUALIZATION_OPTIMIZED=True
VISUALIZE_DETECTED_OUTPUT=False


# --- Helper Function ---
def create_dirs():
    """Creates necessary output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(FILTERED_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(OPTIMIZED_KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    
FLAG_22_POINT=False
ELBOW0=[-348.6, -278.5, -967.0]
ELBOW1=[86.6, -282.6, -960.0]
VISUALIZE_MEDIAPIPE_OUTPUT=False
