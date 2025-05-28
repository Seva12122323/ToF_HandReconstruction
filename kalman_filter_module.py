import numpy as np
from scipy.linalg import solve
from scipy.stats import chi2
import copy
import time
import config 
from utils import butter_lowpass_filter, opt_choose_dep 
import logging


class KalmanFilter:
    """Standard implementation of a linear Kalman filter."""
    def __init__(self, F, H, Q, R, x0, P0, B=None):
        if F is None or H is None or Q is None or R is None or x0 is None or P0 is None: raise ValueError("Provide F, H, Q, R, x0, P0")
        self.F, self.H, self.Q, self.R, self.B = F.copy(), H.copy(), Q.copy(), R.copy(), B.copy() if B is not None else None
        self.P = np.asarray(P0).copy()
        self.x = np.asarray(x0).reshape(-1, 1).copy()
        self.n, self.m = F.shape[1], H.shape[0]
        if self.F.shape != (self.n, self.n): raise ValueError(f"F shape error {self.F.shape} vs {(self.n, self.n)}")
        if self.H.shape != (self.m, self.n): raise ValueError(f"H shape error {self.H.shape} vs {(self.m, self.n)}")
        if self.Q.shape != (self.n, self.n): raise ValueError(f"Q shape error {self.Q.shape} vs {(self.n, self.n)}")
        if self.R.shape != (self.m, self.m): raise ValueError(f"R shape error {self.R.shape} vs {(self.m, self.m)}")
        if self.x.shape != (self.n, 1): raise ValueError(f"x0 shape error {self.x.shape} vs {(self.n, 1)}")
        if self.P.shape != (self.n, self.n): raise ValueError(f"P0 shape error {self.P.shape} vs {(self.n, self.n)}")
        self.y = np.zeros((self.m, 1)); self.S = np.eye(self.m)

    def predict(self, u=None):
        if self.B is not None and u is not None:
            u_vec = np.asarray(u).reshape(-1, 1)
            if self.B.shape[1] != u_vec.shape[0]: raise ValueError("Control u dim mismatch")
            self.x = self.F @ self.x + self.B @ u_vec
        else: self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z_vec = np.asarray(z).reshape(self.m, 1)
        if z_vec.shape != (self.m, 1): raise ValueError(f"Measurement z shape error {z_vec.shape} vs {(self.m, 1)}")
        self.y = z_vec - self.H @ self.x
        self.S = self.H @ self.P @ self.H.T + self.R
        try:
            K = solve(self.S.T, self.H @ self.P.T, assume_a='pos', check_finite=False).T
        except np.linalg.LinAlgError:
            logging.warning("Kalman gain calculation failed (solve). Trying pinv.")
            try:
                K = self.P @ self.H.T @ np.linalg.pinv(self.S)
            except np.linalg.LinAlgError:
                K = np.zeros((self.n, self.m))
                logging.error(f"Kalman gain calculation failed (pinv). Using zero gain.")
        self.x = self.x + K @ self.y
        I = np.eye(self.n)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2.0
        return self.x.copy(), self.P.copy()



def get_kalman_matrices_and_params(filter_config, hand_idx, joint_idx, coord_name, dt):
    """
    Computes Kalman matrices F, H, Q, R, P0, outlier threshold, and state dimension
    based on parameters from the configuration dictionary.
    """
    
    if coord_name not in filter_config['defaults']:
        raise ValueError(f"Default parameters for coordinate '{coord_name}' not found in config.")
    params = copy.deepcopy(filter_config['defaults'][coord_name])
    override_key_wildcard = ('*', joint_idx, coord_name)
    override_key_specific = (hand_idx, joint_idx, coord_name)
    if override_key_wildcard in filter_config['overrides']:
        params.update(filter_config['overrides'][override_key_wildcard])
    if override_key_specific in filter_config['overrides']:
        params.update(filter_config['overrides'][override_key_specific])

    
    model_type = params['model_type']
    sigma_m = params['sigma_measurement']
    sigma_a = params.get('sigma_acceleration', 0.0) # Default if not present (for CP)
    sigma_p_pos = params.get('sigma_process_pos', 0.0) # Default if not present (for CV)
    sigma_i_pos = params['sigma_initial_pos']
    sigma_i_vel = params.get('sigma_initial_vel', 0.0) # Default if not present (for CP)
    p_val = params['p_value_outlier']

    # 3. Compute matrices F, H, Q, R, P0 and threshold
    if model_type == 'CV': # Constant Velocity
        state_dim = 2
        measurement_dim = 1
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        # Process noise Q for CV (using discrete white noise acceleration model)
        q_base = np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]])
        Q = q_base * sigma_a**2
        P0 = np.diag([sigma_i_pos**2, sigma_i_vel**2])
    elif model_type == 'CP': # Constant Position
        state_dim = 1
        measurement_dim = 1
        F = np.array([[1]])
        H = np.array([[1]])
        # Process noise Q for CP
        Q = np.array([[sigma_p_pos**2 * dt]]) 
        P0 = np.array([[sigma_i_pos**2]])
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for ({hand_idx},{joint_idx},{coord_name})")

   
    R = np.array([[sigma_m**2]])


    outlier_threshold = chi2.ppf(1.0 - p_val, df=measurement_dim)

    return F, H, Q, R, P0, outlier_threshold, state_dim, params


class FilteringPipeline:
    """
    Applies Kalman filtering (and optional post-processing) to time-series
    keypoint data.
    """
    def __init__(self, filter_config, frame_rate):
        self.filter_config = filter_config
        self.fs = frame_rate
        if self.fs <= 0:
            raise ValueError("Frame rate (fs) must be positive.")
        self.dt = 1.0 / self.fs
        self.filters = {} 
        self.outlier_thresholds = {}
        self.state_dims = {}
        self.initial_states = {}
        self.is_initialized = False
        logging.info(f"FilteringPipeline initialized with dt={self.dt:.4f}s.")

    def _initialize_filters(self, initial_data):
        """Initializes Kalman filters based on the first frame(s) of data."""
        if initial_data is None or initial_data.shape[0] < 1:
            raise ValueError("Initial data is required to initialize filters.")

        num_frames, num_hands, num_landmarks, num_coords = initial_data.shape
        if num_hands != config.NUM_HANDS or num_landmarks != config.NUM_LANDMARKS_MEDIAPIPE or num_coords != config.NUM_COORDS:
             logging.warning(f"Initial data shape mismatch: {initial_data.shape}. Expected (_, {config.NUM_HANDS}, {config.NUM_LANDMARKS_MEDIAPIPE}, {config.NUM_COORDS}).")
             

        logging.info(f"Initializing {num_hands * num_landmarks * num_coords} Kalman filters...")
        init_errors = 0

        for i in range(num_hands):
            for j in range(num_landmarks):
                for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
                    filter_key = (i, j, coord_name)
                    try:
                        F, H, Q, R, P0, threshold, state_dim, _ = get_kalman_matrices_and_params(
                            self.filter_config, i, j, coord_name, self.dt
                        )
                        self.state_dims[filter_key] = state_dim
                        self.outlier_thresholds[filter_key] = threshold

                        # Determine initial state x0
                        pos0 = initial_data[0, i, j, coord_idx]
                        # Handle NaN in initial position
                        if np.isnan(pos0):
                            logging.warning(f"NaN found in initial position for {filter_key}. Using 0.")
                            pos0 = 0.0

                        vel0 = 0.0
                        if num_frames > 1 and state_dim == 2: 
                            pos1 = initial_data[1, i, j, coord_idx]
                            if not np.isnan(pos0) and not np.isnan(pos1):
                                vel0 = (pos1 - pos0) / self.dt
                            elif np.isnan(pos0) and not np.isnan(pos1):
                                pos0 = pos1 
                                vel0 = 0.0
                           

                        # Create initial state vector x0
                        if state_dim == 1: x0 = np.array([pos0])
                        elif state_dim == 2: x0 = np.array([pos0, vel0])
                        else: raise ValueError(f"Unsupported state dimension {state_dim}")

                        self.initial_states[filter_key] = x0.copy()
                        self.filters[filter_key] = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

                    except Exception as e:
                        logging.error(f"ERROR initializing filter for {filter_key}: {e}", exc_info=True)
                        self.filters[filter_key] = None
                        self.outlier_thresholds[filter_key] = None
                        self.state_dims[filter_key] = 0
                        self.initial_states[filter_key] = np.array([np.nan]) # Indicate error
                        init_errors += 1

        if init_errors > 0:
            logging.warning(f"{init_errors} filters failed to initialize.")
        else:
            logging.info("All Kalman filters initialized successfully.")
        self.is_initialized = True


    def apply_filter(self, all_frames_data_raw):
        """
        Applies the initialized Kalman filters to the entire dataset.

        Args:
            all_frames_data_raw (np.ndarray): Raw keypoint data, shape
                                             (num_frames, num_hands, num_landmarks, 3).

        Returns:
            np.ndarray: Filtered keypoint data including the 22nd point,
                        shape (num_frames, num_hands, num_landmarks+1, 3).
        """
        if not self.is_initialized:
             
             self._initialize_filters(all_frames_data_raw)
             if not self.is_initialized: 
                  raise RuntimeError("Filters must be initialized before applying.")

        num_frames, num_hands, num_landmarks, num_coords = all_frames_data_raw.shape
        logging.info(f"Starting Kalman filtering for {num_frames} frames...")


        data_to_filter = all_frames_data_raw.copy()
        if config.APPLY_Z_OPTIMIZATION_PRE_FILTER:
            logging.warning("Pre-filter Z optimization is enabled but not implemented in this refactoring yet. Using raw Z data.")

        max_state_dim = max(self.state_dims.values()) if self.state_dims else 1
        filtered_states_array = np.full((num_frames, num_hands, num_landmarks, num_coords, max_state_dim), np.nan)

        
        for i in range(num_hands):
            for j in range(num_landmarks):
                for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
                    filter_key = (i, j, coord_name)
                    if self.filters.get(filter_key) is not None:
                         initial_state = self.initial_states[filter_key]
                         dim = len(initial_state)
                         filtered_states_array[0, i, j, coord_idx, :dim] = initial_state
                    

        # Main Filtering Loop
        start_kalman_time = time.time()
        skipped_update_nan = 0
        skipped_update_outlier = 0
        update_errors = 0

        for t in range(1, num_frames): # Start from the second frame
            if t % max(1, num_frames // 10) == 0:
                logging.info(f"  Filtering frame {t+1}/{num_frames}")

            for i in range(num_hands):
                for j in range(num_landmarks):
                    for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
                        filter_key = (i, j, coord_name)
                        kf = self.filters.get(filter_key)

                        if kf is None: continue 

                        # Prediction
                        try:
                            predicted_state = kf.predict()
                        except Exception as e:
                            logging.error(f"ERROR in Kalman predict t={t}, key={filter_key}: {e}")
  
                            prev_state = filtered_states_array[t-1, i, j, coord_idx, :kf.n]
                            if np.isnan(prev_state).any(): prev_state = np.zeros(kf.n) 
                            predicted_state = prev_state.reshape(-1, 1)
                            kf.x = predicted_state 


                        measurement = data_to_filter[t, i, j, coord_idx]
                        measurement_is_nan = np.isnan(measurement)

                        # Update
                        final_state = predicted_state 
                        if measurement_is_nan:
                            skipped_update_nan += 1
                        else:
                            threshold = self.outlier_thresholds.get(filter_key)
                            is_outlier = False
                            if threshold is not None:
                                try:
                                    y = measurement - kf.H @ predicted_state 
                                    S_val = kf.S[0,0] 
                                    if S_val < 1e-9: 
                                        is_outlier = True
                                    else:
                                        mahalanobis_sq = (y**2 / S_val)[0,0]
                                        if mahalanobis_sq > threshold:
                                            is_outlier = True
                                except (np.linalg.LinAlgError, ValueError, IndexError, ZeroDivisionError, AttributeError) as e:
                                    is_outlier = True # Treat as outlier if check fails

                            if is_outlier:
                                skipped_update_outlier += 1
                            else:
                                
                                try:
                                    updated_state, _ = kf.update(measurement)
                                    final_state = updated_state # Use updated state
                                except Exception as e:
                                    logging.error(f"ERROR in Kalman update t={t}, key={filter_key}: {e}")
                                    update_errors += 1
                                    # final_state remains predicted_state

                        # Store the final state for this time step
                        dim = len(final_state)
                        filtered_states_array[t, i, j, coord_idx, :dim] = final_state.flatten()

        end_kalman_time = time.time()
        logging.info(f"Kalman filtering finished in {end_kalman_time - start_kalman_time:.2f} seconds.")
        if skipped_update_nan > 0: logging.info(f"Skipped {skipped_update_nan} updates (NaN measurement).")
        if skipped_update_outlier > 0: logging.info(f"Skipped {skipped_update_outlier} updates (outlier detected).")
        if update_errors > 0: logging.warning(f"Encountered {update_errors} update errors.")

        
        logging.info("Assembling final positions from filtered states...")
        final_positions = filtered_states_array[:, :, :, :, 0] # Shape: (frames, hands, landmarks, coords)

        # Post-filter Butterworth
        if config.APPLY_BUTTERWORTH_POST_FILTER:
            logging.info("Applying Butterworth low-pass filter AFTER Kalman (Z coordinate)...")
            butter_errors = 0
            for i in range(num_hands):
                for j in range(num_landmarks):
                    z_coords_kalman = final_positions[:, i, j, 2]
                    # Apply filter only if enough data points exist
                    if len(z_coords_kalman) > config.BUTTERWORTH_ORDER * 3:
                        z_coords_filtered_butter = butter_lowpass_filter(
                            z_coords_kalman,
                            config.BUTTERWORTH_CUTOFF,
                            config.BUTTERWORTH_FS,
                            order=config.BUTTERWORTH_ORDER
                        )
                        if z_coords_filtered_butter.shape == z_coords_kalman.shape:
                            final_positions[:, i, j, 2] = z_coords_filtered_butter
                        else:
                            butter_errors +=1
                            logging.warning(f"Butterworth output shape mismatch for hand {i}, landmark {j}.")


            if butter_errors > 0: logging.warning(f"Butterworth filtering encountered {butter_errors} errors.")
            logging.info("Butterworth post-filtering applied to Z coordinate.")
        else:
            logging.info("Skipping Butterworth post-filtering.")

        # Add 22nd Point 
        logging.info("Adding the 22nd point based on filtered wrist position...")
        keypoints_data_final = np.full((num_frames, num_hands, config.NUM_LANDMARKS_EXTENDED, num_coords), np.nan)
        keypoints_data_final[:, :, :num_landmarks, :] = final_positions

        
        wrist_pos = final_positions[:, :, 0, :] 
        new_point_pos = wrist_pos.copy()
        new_point_pos[:, :, 1] += config.Y_OFFSET_FOR_NEW_POINT 
        if config.FLAG_22_POINT:
            keypoints_data_final[:, 0, num_landmarks, :] = config.ELBOW0
            keypoints_data_final[:, 1, num_landmarks, :] = config.ELBOW1
        else:
            keypoints_data_final[:, :, num_landmarks, :] = new_point_pos 

        logging.info(f"Final data prepared with {config.NUM_LANDMARKS_EXTENDED} landmarks. Shape: {keypoints_data_final.shape}")

        return keypoints_data_final