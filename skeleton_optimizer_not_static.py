import numpy as np
from scipy.optimize import minimize
import config 
import logging
import time

def objective_total_error(x_flat, detected_landmarks, num_landmarks, target_bone_lengths_dict,
                           weight_pose_sq, weight_pose_abs, weight_bone_error):
    """
    Целевая функция, включающая:
    1. Ошибку позиционирования (отклонение от detected_landmarks).
    2. Ошибку длин костей (отклонение от target_bone_lengths_dict).
    """
    landmarks_opt = x_flat.reshape((num_landmarks, 3))


    pose_error_component = 0.0
    if detected_landmarks is not None: 
        if landmarks_opt.shape != detected_landmarks.shape:
            logging.error(f"Shape mismatch: opt {landmarks_opt.shape} vs det {detected_landmarks.shape}")
           
            return np.inf 

        diff = landmarks_opt - detected_landmarks
        
        valid_mask_detected = ~np.isnan(detected_landmarks).any(axis=1)
        
        if np.any(valid_mask_detected):
            diff_valid = diff[valid_mask_detected]
            
            error_sq = weight_pose_sq * np.sum(diff_valid**2)
            error_abs = weight_pose_abs * np.sum(np.sqrt(np.sum(diff_valid**2, axis=1) + 1e-9))
            pose_error_component = error_sq + error_abs


    # Bone length errorr
    bone_error_component = 0.0
    if target_bone_lengths_dict: 
        num_bones_calculated = 0
        for (start_idx, end_idx), target_len in target_bone_lengths_dict.items():
            if not (0 <= start_idx < num_landmarks and \
                    0 <= end_idx < num_landmarks and \
                    isinstance(target_len, (int, float)) and \
                    not np.isnan(target_len) and target_len > 1e-6): 
                continue
            
            p_start = landmarks_opt[start_idx]
            p_end = landmarks_opt[end_idx]


            if np.isnan(p_start).any() or np.isnan(p_end).any():
                bone_error_component += 1e12 # if NaN huge errorr
                continue

            current_len = np.linalg.norm(p_end - p_start)
            bone_error_component += (current_len - target_len)**2 
            num_bones_calculated +=1

    total_error = pose_error_component + weight_bone_error * bone_error_component
    
    if np.isnan(total_error):
        logging.warning(f"NaN in objective_total_error! PoseComp={pose_error_component}, BoneComp={bone_error_component}")
        return np.inf # Optimizer avoid NaN
        
    return total_error


def optimize_skeleton(initial_landmarks, target_bone_lengths, detected_landmarks, k_iter=1.0):
    """
    Оптимизирует положение скелета, минимизируя ошибку позиционирования
    и ошибку длин костей, обе включены в целевую функцию.
    """
    if initial_landmarks is None:
        logging.error("Initial_landmarks is None. Skipping optimization.")
        return initial_landmarks 
    
    num_landmarks = initial_landmarks.shape[0]

    if not isinstance(initial_landmarks, np.ndarray) or initial_landmarks.ndim != 2 or initial_landmarks.shape[1] != 3:
        logging.error(f"Invalid initial_landmarks shape: {initial_landmarks.shape}. Expected ({num_landmarks}, 3)")
        return initial_landmarks.copy() if isinstance(initial_landmarks, np.ndarray) else None
        
    if detected_landmarks is not None and (not isinstance(detected_landmarks, np.ndarray) or detected_landmarks.shape != (num_landmarks, 3)):
        logging.error(f"Invalid detected_landmarks shape: {detected_landmarks.shape}. Expected ({num_landmarks}, 3) or None")
        return initial_landmarks.copy()

    initial_values_flat = initial_landmarks.flatten()
    if np.isnan(initial_values_flat).any():
        logging.warning("NaNs found in initial_landmarks for optimizer. Replacing with 0.0 for those specific NaNs.")
        initial_values_flat = np.nan_to_num(initial_values_flat, nan=0.0)

    if detected_landmarks is not None and np.all(np.isnan(detected_landmarks)):
        logging.warning("All detected_landmarks are NaN. Pose error component will be zero if weights are >0.")


    weight_pose_sq = config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ if hasattr(config, 'OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ') else 0.1
    weight_pose_abs = config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS if hasattr(config, 'OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS') else 0.5
    
    
    weight_bone_error = config.OPTIMIZATION_BONE_LENGTH_ERROR_WEIGHT if hasattr(config, 'OPTIMIZATION_BONE_LENGTH_ERROR_WEIGHT') else 10.0 

    logging.info(f"Optimization weights: pose_sq={weight_pose_sq}, pose_abs={weight_pose_abs}, bone_err={weight_bone_error}")

    base_maxiter = config.OPTIMIZATION_MAX_ITER if hasattr(config, 'OPTIMIZATION_MAX_ITER') else 100 # Базовое число итераций
    
    optimization_params = {
        'method': config.OPTIMIZATION_METHOD, 
        'options': {
            'disp': False,  
            'maxiter': int(base_maxiter * k_iter),
            'ftol': 1e-4, 
            'gtol': 1e-3  
        }
    }
    logging.info(f"Starting optimization with method {optimization_params['method']} for {num_landmarks} landmarks...")
    start_time = time.time()

    try:
        result = minimize(
            objective_total_error,
            initial_values_flat,
            args=(detected_landmarks, num_landmarks, target_bone_lengths, 
                  weight_pose_sq, weight_pose_abs, weight_bone_error), 
            **optimization_params
        )

        elapsed_time = time.time() - start_time
        if result.success:
            optimized_landmarks = result.x.reshape((num_landmarks, 3))
            if np.isnan(optimized_landmarks).any():
                logging.error("Optimized landmarks contain NaN. Reverting to initial guess.")
                return initial_landmarks.copy()
            
            
            change_norm = np.linalg.norm(initial_landmarks.flatten() - optimized_landmarks.flatten())
            logging.info(f"Optimization SUCCEEDED. Iterations: {result.nit}. Func Evals: {result.nfev}. Time: {elapsed_time:.2f}s. Change norm: {change_norm:.3f}. Message: {result.message}")
            return optimized_landmarks
        else:
            logging.warning(f"Optimization FAILED. Iterations: {result.nit}. Func Evals: {result.nfev}. Time: {elapsed_time:.2f}s. Message: {result.message}")
            return initial_landmarks.copy()

    except Exception as e:
        logging.error(f"Optimization error: {str(e)}", exc_info=True)
        return initial_landmarks.copy()

