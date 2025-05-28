import numpy as np
from scipy.optimize import minimize
import config  
import logging
import time


# Errorr func
def objective_pose_error(x_flat, detected_landmarks, num_landmarks):
    """
    Целевая функция для минимизации: ошибка позиционирования.
    Рассчитывает взвешенную ошибку между текущими (оптимизируемыми)
    и обнаруженными позициями лендмарков.

    Args:
        x_flat (np.ndarray): Плоский массив текущих координат (num_landmarks * 3).
        detected_landmarks (np.ndarray): Обнаруженные координаты (num_landmarks, 3).
        num_landmarks (int): Количество лендмарков.

    Returns:
        float: Значение ошибки позиционирования.
    """
    landmarks_opt = x_flat.reshape((num_landmarks, 3))

    if landmarks_opt.shape != detected_landmarks.shape:
         logging.error(f"Shape mismatch inside objective function: opt{landmarks_opt.shape} vs det{detected_landmarks.shape}")
         return np.inf

    diff = landmarks_opt - detected_landmarks

    valid_mask = ~np.isnan(detected_landmarks).any(axis=1)
    if not np.any(valid_mask):
        logging.warning("No valid detected landmarks to compare in objective function.")
        return 0.0

    diff_valid = diff[valid_mask]

    error_sq = config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_SQ * np.sum(diff_valid**2)

    diff_norm_sq = np.sum(diff_valid**2, axis=1)
    epsilon = 1e-9
    error_abs = config.OPTIMIZATION_LANDMARK_ERROR_WEIGHT_ABS * np.sum(np.sqrt(diff_norm_sq + epsilon))

    total_pose_error = error_sq + error_abs

    if np.isnan(total_pose_error) or np.isinf(total_pose_error):
        logging.warning(f"NaN or Inf detected in objective function result. Error SQ: {error_sq}, ABS: {error_abs}")
        return np.inf

    return total_pose_error


# Constraint
def constraint_bone_length(x_flat, start_idx, end_idx, target_length, num_landmarks):
    """
    Функция ограничения для одной кости (для scipy.optimize).
    Возвращает значение, которое оптимизатор будет пытаться сделать равным нулю.
    (Текущая длина^2 - Целевая длина^2)

    Args:
        x_flat (np.ndarray): Плоский массив текущих координат (num_landmarks * 3).
        start_idx (int): Индекс начальной точки кости.
        end_idx (int): Индекс конечной точки кости.
        target_length (float): Требуемая длина кости (из config.BONE_LENGTHS).
        num_landmarks (int): Количество лендмарков.

    Returns:
        float: `current_distance_squared - target_length_squared`.
    """
    landmarks_opt = x_flat.reshape((num_landmarks, 3))

    if start_idx >= num_landmarks or end_idx >= num_landmarks:
        logging.error(f"Invalid indices ({start_idx}, {end_idx}) in constraint function for {num_landmarks} landmarks.")
        return np.inf

    p_start = landmarks_opt[start_idx]
    p_end = landmarks_opt[end_idx]

    if np.isnan(p_start).any() or np.isnan(p_end).any():
        logging.warning(f"NaN found in optimized landmarks for constraint ({start_idx},{end_idx}).")
        return np.inf

    current_dist_sq = np.sum((p_start - p_end)**2)
    target_length_sq = target_length**2

    if current_dist_sq < 0:
        logging.warning(f"Negative squared distance encountered for constraint ({start_idx},{end_idx}).")
        current_dist_sq = 0

    result = current_dist_sq - target_length_sq
    if np.isnan(result) or np.isinf(result):
       logging.warning(f"NaN or Inf detected in constraint function result for ({start_idx},{end_idx}). DistSq: {current_dist_sq}, TargetSq: {target_length_sq}")
       return np.inf

    return result

def create_constraints_list(target_bone_lengths, num_landmarks):
    """
    Создает список словарей ограничений для `scipy.optimize.minimize`.

    Args:
        target_bone_lengths (dict): Словарь с целевыми длинами костей (из config).
        num_landmarks (int): Общее количество лендмарков в скелете.

    Returns:
        list: Список словарей ограничений.
    """
    constraints = []
    if not target_bone_lengths:
        logging.warning("Empty target_bone_lengths provided, no constraints will be created.")
        return []

    for (start, end), target_length in target_bone_lengths.items():
        # Validation before adding
        if not isinstance(start, int) or not isinstance(end, int):
             logging.warning(f"Skipping bone constraint ({start},{end}): indices must be integers.")
             continue
        if start >= num_landmarks or end >= num_landmarks or start < 0 or end < 0:
            logging.warning(f"Skipping bone constraint ({start},{end}): index out of bounds [0, {num_landmarks-1}].")
            continue
        if start == end:
             logging.warning(f"Skipping bone constraint ({start},{end}): start and end indices are the same.")
             continue
        if not isinstance(target_length, (int, float)) or np.isnan(target_length):
            logging.warning(f"Skipping bone constraint ({start},{end}): target length is not a valid number (NaN or type {type(target_length)}).")
            continue
        if target_length < 0:
            logging.warning(f"Skipping bone constraint ({start},{end}): target length is negative ({target_length}).")
            continue
        if target_length == 0:
             logging.warning(f"Skipping bone constraint ({start},{end}): target length is zero.")
             continue

        constraints.append({
            'type': 'eq',
            'fun': constraint_bone_length,
            'args': (start, end, target_length, num_landmarks)
        })

    logging.info(f"Created {len(constraints)} bone length equality constraints.")
    return constraints

#Optimization function
def optimize_skeleton(initial_landmarks, target_bone_lengths, detected_landmarks, k_iter=1.0):
    """
    Оптимизирует положение лендмарков, минимизируя ошибку позы
    при строгом соблюдении заданных длин костей.

    Args:
        initial_landmarks (np.ndarray): Начальное приближение (num_landmarks, 3).
        target_bone_lengths (dict): Словарь {(start_idx, end_idx): length},
                                    обычно берется из config.BONE_LENGTHS.
        detected_landmarks (np.ndarray): Обнаруженные/целевые координаты (num_landmarks, 3).
        k_iter (float): Множитель для максимального числа итераций.

    Returns:
        np.ndarray: Оптимизированные координаты (num_landmarks, 3) или
                    initial_landmarks, если оптимизация не удалась или входные данные неверны.
    """
    # Check data
    if initial_landmarks is None or detected_landmarks is None or target_bone_lengths is None:
        logging.error("optimize_skeleton: Missing input data.")
        return initial_landmarks

    if not isinstance(initial_landmarks, np.ndarray) or initial_landmarks.ndim != 2 or initial_landmarks.shape[1] != 3:
        logging.error(f"optimize_skeleton: Invalid initial_landmarks shape/type: {initial_landmarks.shape if isinstance(initial_landmarks, np.ndarray) else type(initial_landmarks)}. Expected (N, 3).")
        return initial_landmarks

    num_landmarks = initial_landmarks.shape[0]

    if not isinstance(detected_landmarks, np.ndarray) or detected_landmarks.shape != (num_landmarks, 3):
         logging.error(f"optimize_skeleton: Invalid detected_landmarks shape/type: {detected_landmarks.shape if isinstance(detected_landmarks, np.ndarray) else type(detected_landmarks)}. Expected ({num_landmarks}, 3).")
         return initial_landmarks

    if not isinstance(target_bone_lengths, dict):
        logging.error(f"optimize_skeleton: Invalid target_bone_lengths type: {type(target_bone_lengths)}. Expected dict.")
        return initial_landmarks

    if num_landmarks == 0:
        logging.warning("optimize_skeleton: No landmarks provided (num_landmarks=0).")
        return initial_landmarks

    # Preparing for optimization
    if np.isnan(initial_landmarks).any():
         logging.warning("NaNs found in initial guess. Replacing with zeros.")
         initial_values_flat = np.nan_to_num(initial_landmarks, nan=0.0).flatten()
    else:
        initial_values_flat = initial_landmarks.flatten()

    if np.isnan(detected_landmarks).any():
        logging.warning("NaNs found in detected_landmarks. Objective function will ignore them.")
        if np.all(np.isnan(detected_landmarks)):
             logging.error("All detected_landmarks are NaN. Cannot optimize.")
             return initial_landmarks

    # Constraints list making
    constraints = create_constraints_list(target_bone_lengths, num_landmarks)
    if not constraints:
        logging.error("No valid constraints could be created from the provided bone lengths. Returning initial guess.")
        return initial_landmarks

    logging.info(f"Starting constrained optimization for {num_landmarks} landmarks...")
    optimization_start_time = time.time()

    try:
        max_iter = int(config.OPTIMIZATION_MAX_ITER * k_iter)
        if max_iter <= 0: max_iter = 10
        logging.debug(f"Using optimizer: {config.OPTIMIZATION_METHOD}, Max iterations: {max_iter}")

        result = minimize(
            objective_pose_error,
            initial_values_flat,
            args=(detected_landmarks, num_landmarks),
            method=config.OPTIMIZATION_METHOD,        
            constraints=constraints,                 
            options={
                'disp': False,
                'maxiter': max_iter,
                'ftol': 1e-2
                }
        )
        optimization_end_time = time.time()
        duration = optimization_end_time - optimization_start_time

        # Post-Processing
        if result.success:
            optimized_landmarks = result.x.reshape((num_landmarks, 3))
            if np.isnan(optimized_landmarks).any():
                 logging.error(f"Optimization succeeded (scipy), but result contains NaNs! Message: {result.message}. Returning initial guess.")
                 return initial_landmarks
            logging.info(f"Optimization successful in {duration:.3f}s. Final error: {result.fun:.4f}, Iterations: {result.nit}")
            return optimized_landmarks
        else:
            logging.warning(f"Optimization failed after {duration:.3f}s. Message: {result.message}. Status: {result.status}. Iterations: {result.nit}. Final error: {result.fun:.4f}")
            return initial_landmarks

    except ValueError as ve:
         logging.error(f"ValueError during optimization setup or execution: {ve}", exc_info=True)
         return initial_landmarks
    except Exception as e:
        logging.error(f"Unexpected error during optimization: {e}", exc_info=True)
        return initial_landmarks

# Check constraints
def check_final_constraints(landmarks, target_bone_lengths):
    """Проверяет, насколько хорошо выполнены ограничения после оптимизации."""
    if landmarks is None or target_bone_lengths is None or not isinstance(landmarks, np.ndarray) or not isinstance(target_bone_lengths, dict):
        return
    num_landmarks = landmarks.shape[0]
    logging.debug("--- Checking final bone lengths ---")
    max_abs_error = 0
    total_sq_error = 0
    count = 0
    for (start, end), target_length in target_bone_lengths.items():
         if isinstance(start, int) and isinstance(end, int) and \
            0 <= start < num_landmarks and 0 <= end < num_landmarks and \
            start != end and isinstance(target_length, (int, float)) and \
            not np.isnan(target_length) and target_length > 0:

            p_start = landmarks[start]
            p_end = landmarks[end]
            if np.isnan(p_start).any() or np.isnan(p_end).any():
                 logging.warning(f"  Constraint check skipped for ({start},{end}): NaN in optimized coords.")
                 continue

            current_dist = np.linalg.norm(p_start - p_end)
            abs_error = abs(current_dist - target_length)
            logging.debug(f"  Bone ({start:2d}, {end:2d}): Target={target_length:6.2f}, Actual={current_dist:6.2f}, Abs Error={abs_error:.4f}")
            max_abs_error = max(max_abs_error, abs_error)
            total_sq_error += abs_error**2
            count += 1
    if count > 0:
        rms_error = np.sqrt(total_sq_error / count)
        logging.debug(f"Max absolute bone length error: {max_abs_error:.4f}")
        logging.debug(f"RMS bone length error: {rms_error:.4f}")
    else:
        logging.debug("No valid constraints to check.")
    logging.debug("--- End of constraint check ---")



