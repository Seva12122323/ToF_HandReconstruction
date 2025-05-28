import numpy as np
import matplotlib.pyplot as plt # Импортируем matplotlib
from matplotlib import ticker # !!! ДОБАВЛЯЕМ ИМПОРТ TICKER !!!
# from mpl_toolkits.mplot3d import Axes3D # Этот импорт больше не нужен
import math
from scipy.signal import butter, lfilter
import os # Добавим для проверки файлов

# --- Функции чтения и фильтрации (без изменений) ---
def read_hand_skeleton_data(file_path, thickness_array=None):
    """
    Reads 3D hand keypoints from a file and applies thickness adjustments
    to the z-coordinate *during* the reading process.
    """
    keypoints_list = []
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
        (0, 17), (0, 21)  # Wrist
    ]
    num_landmarks = 22 # Ожидаемое количество точек

    # Переделка логики толщины: применяем к точкам, а не связям
    if thickness_array is not None:
        if len(thickness_array) != num_landmarks:
            raise ValueError(f"Thickness array length ({len(thickness_array)}) must be {num_landmarks}")
        thickness_adjustment_enabled = True
    else:
        thickness_adjustment_enabled = False
        # print("Warning: No thickness array provided. Skipping thickness adjustment.")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            current_hand = []
            landmark_index = 0
            for line in lines:
                if line.startswith("Hand"):
                    if current_hand:
                        # Дополняем нулями, если точек меньше 21
                        while len(current_hand) < num_landmarks:
                             current_hand.append([0.0, 0.0, 0.0])
                        keypoints_list.append(np.array(current_hand[:num_landmarks])) # Берем только 21 точку
                    current_hand = []
                    landmark_index = 0
                elif line.startswith("  Landmark"):
                    if landmark_index >= num_landmarks: continue # Пропускаем лишние точки

                    try:
                        parts = line.split(":")
                        coords_str = parts[1].strip()
                        coords = {}
                        for item in coords_str.split(", "):
                            k, v = item.split("=")
                            coords[k] = float(v)
                        # Используем get для безопасности + инвертируем Z и масштабируем
                        x = coords.get('x', 0.0)
                        y = coords.get('y', 0.0)
                        z = -coords.get('z', 0.0) 

                        # Применяем толщину к текущей точке
                        if thickness_adjustment_enabled:
                            z -= thickness_array[landmark_index]

                        current_hand.append([x, y, z])
                        landmark_index += 1
                    except Exception as e:
                         print(f"Error parsing line: '{line.strip()}' in file {file_path}. Error: {e}. Appending [0,0,0].")
                         current_hand.append([0.0, 0.0, 0.0]) # Добавляем ноль при ошибке
                         landmark_index += 1


            if current_hand:
                 # Дополняем нулями последнюю руку, если нужно
                 while len(current_hand) < num_landmarks:
                      current_hand.append([0.0, 0.0, 0.0])
                 keypoints_list.append(np.array(current_hand[:num_landmarks]))

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Returning default zero hands.")
        # Возвращаем структуру с нулями, если файл не найден
        return [np.zeros((num_landmarks, 3)), np.zeros((num_landmarks, 3))]
    except Exception as e:
        print(f"An unexpected error occurred reading {file_path}: {e}. Returning default zero hands.")
        return [np.zeros((num_landmarks, 3)), np.zeros((num_landmarks, 3))]


    # Гарантируем возврат списка из 2 рук (numpy array 21x3)
    final_list = []
    for hand in keypoints_list[:2]: # Берем не больше 2 рук
        if hand.shape == (num_landmarks, 3):
            final_list.append(hand)
    while len(final_list) < 2: # Дополняем нулями до 2 рук, если нужно
        final_list.append(np.zeros((num_landmarks, 3)))

    return final_list

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Проверка граничных случаев для стабильности
    if normal_cutoff >= 1.0: normal_cutoff = 0.999
    if normal_cutoff <= 0.0: normal_cutoff = 0.001
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
     # Проверка на достаточную длину данных
    if len(data) <= order * 3:
        # print(f"Warning: Data length ({len(data)}) too short for filter order ({order}). Returning original data.")
        return data
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

# --- Основной блок ---
if __name__ == "__main__":
    # --- Параметры ---
    num_frames = 575
    start_frame_analysis = 0 # С какого кадра начинать анализ для Delta
    start_frame_angle_analysis = 0 # С какого кадра начинать анализ для углов
    fs = 30.0  # Частота дискретизации (кадров в секунду)
    dt = 1.0 / fs # Шаг по времени (секунд на кадр)
    cutoff = 4.55 # Частота среза фильтра
    filter_order = 3 # Порядок фильтра

    # --- Массив толщин (без изменений) ---
    thickness_array = np.array([
        10, 5, 5, 5, 5,  # Wrist + Thumb
         8, 6, 6, 6,     # Index
         8, 6, 6, 6,     # Middle
         8, 6, 6, 6,     # Ring
         8, 6, 6, 6 , 3     # Pinky
    ])
    # thickness_array = None

    # --- Загрузка данных (без изменений) ---
    print("Загрузка данных...")
    keypoints_data = []
    keypoints_data_aligned = []
    missing_files = 0
    for k in range(num_frames):
        file_path = f"D:\\DyplomaProject\\hand_pose_pipeline\\output\\filtered_keypoints\\{k}_filtered_keypoints.txt"
        # file_path_aligned = f"{k + 1}_hand_keypoints_aligned.txt" # Предполагаем, что он нужен

        if not os.path.exists(file_path) : # or not os.path.exists(file_path_aligned):
             keypoints_data.append([np.zeros((21, 3)), np.zeros((21, 3))])
             # keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])
             missing_files += 1
             continue

        keypoints_raw = read_hand_skeleton_data(file_path, thickness_array)
        keypoints_data.append(keypoints_raw)

        # if os.path.exists(file_path_aligned):
        #      keypoints_raw_aligned = read_hand_skeleton_data(file_path_aligned, thickness_array)
        #      keypoints_data_aligned.append(keypoints_raw_aligned)
        # else:
        #      keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])

    if missing_files > 0: print(f"Предупреждение: Пропущено {missing_files} кадров.")

    keypoints_data = np.array(keypoints_data)
    # keypoints_data_aligned = np.array(keypoints_data_aligned)
    print(f"Данные загружены. Форма основного файла: {keypoints_data.shape}")

    # --- Фильтрация данных (без изменений) ---
    print("Фильтрация данных...")
    keypoints_data_filtered = np.zeros_like(keypoints_data)
    for hand_idx in range(keypoints_data.shape[1]):
        for landmark_idx in range(keypoints_data.shape[2]):
            for coord_idx in range(keypoints_data.shape[3]):
                time_series = keypoints_data[:, hand_idx, landmark_idx, coord_idx]
                if np.any(time_series):
                    keypoints_data_filtered[:, hand_idx, landmark_idx, coord_idx] = butter_lowpass_filter(time_series, cutoff, fs, order=filter_order)
                else:
                    keypoints_data_filtered[:, hand_idx, landmark_idx, coord_idx] = time_series
    print("Фильтрация завершена.")

    # --- Расчет метрик (без изменений) ---
    print("Расчет метрик...")
    right_hand_idx = 1
    intervals = [[20, 200], [230, 374], [405,550]]
   

    angle_diff = np.array([])
    min_indices = []
    mean_min_dist = np.nan
    delta_x, delta_y, delta_z = np.array([]), np.array([]), np.array([])
    metric_delta = np.nan
    angles = np.array([])
    angles_third_derivative = np.array([])
    metric_smoothness = np.nan

    # 1. Метрика: Мин расстояние
    if keypoints_data_filtered.shape[1] > right_hand_idx and keypoints_data_filtered.shape[0] > 0:
        elbow = keypoints_data_filtered[:, right_hand_idx, 21] 
        core_of_hand = keypoints_data_filtered[:, right_hand_idx, 0]
        center_of_hand = (keypoints_data_filtered[:, right_hand_idx, 10]+keypoints_data_filtered[:, right_hand_idx, 14])/2
    #     print(center_of_hand)
    #     angle_diff = [np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))) 
    #          for a, b in zip(
    #              (elbow - core_of_hand) / np.linalg.norm(elbow - core_of_hand),
    #              (center_of_hand - core_of_hand) / np.linalg.norm(center_of_hand - core_of_hand)
    #          )]
    #     list_metrics_min_angle = []
    #     list_metrics_max_angle = []
    #     min_indices = []
    #     for start, end in intervals:
    #         start_clamped = max(0, start)
    #         end_clamped = min(len(angle_diff), end)
    #         if start_clamped < end_clamped:
    #             interval_data = angle_diff[start_clamped:end_clamped]
    #             if len(interval_data) > 0:
    #                 min_val = np.min(interval_data)
    #                 max_val = np.max(interval_data)
    #                 max_idx_local = np.argmax(interval_data)
    #                 min_idx_local = np.argmin(interval_data)
    #                 list_metrics_min_angle.append(min_val)
    #                 list_metrics_max_angle.append(max_val)
    #                 min_indices.append(start_clamped + min_idx_local)
    #                 min_indices.append(start_clamped + max_idx_local)
    #             else: list_metrics_min_angle.append(np.nan)
    #         else: list_metrics_min_angle.append(np.nan)
    #     valid_metrics_min= [m for m in list_metrics_min_angle if not np.isnan(m)]
    #     valid_metrics_max = [m for m in list_metrics_max_angle if not np.isnan(m)]
    #     mean_min_dist = np.mean(valid_metrics_min) if valid_metrics_min else np.nan
    #     print(f"Метрика 'Мин расстояние до б пальца': {mean_min_dist:.2f}")
    #     mean_max_dist = np.mean(valid_metrics_max) if valid_metrics_max else np.nan
    #     print(f"Метрика 'Мин расстояние до б пальца': {mean_min_dist:.2f}")
    # else: print("Ошибка: Правая рука не найдена или недостаточно данных для расчета метрики расстояния.")

    # 2. Метрика: Delta (Дрожание)
    if keypoints_data_filtered.shape[1] > right_hand_idx and len(keypoints_data_filtered) > start_frame_analysis:
        wrist_coords = keypoints_data_filtered[start_frame_analysis:, right_hand_idx, 0, :]
        delta_z, delta_y, delta_x = wrist_coords[:, 2], wrist_coords[:, 1], wrist_coords[:, 0]
        if len(delta_x) > 1:
            std_z, std_y, std_x = np.std(delta_z), np.std(delta_y), np.std(delta_x)
            metric_delta = np.sqrt(std_z**2 + std_y**2 + std_x**2)
            print(f"Метрика 'Delta (Дрожание)': {metric_delta:.2f}")
        else:
            print("Предупреждение: Недостаточно данных после start_frame_analysis для расчета Delta.")
            metric_delta = np.nan
            delta_x, delta_y, delta_z = np.array([]), np.array([]), np.array([])
    else: print("Ошибка: Правая рука не найдена или недостаточно кадров для расчета метрики Delta.")

    # 3. Метрика: Коэф плавности (Jerk)
    if keypoints_data_filtered.shape[1] > right_hand_idx and len(keypoints_data_filtered) > start_frame_angle_analysis:
        try:
            points_for_angle = keypoints_data_filtered[start_frame_angle_analysis:, right_hand_idx, [4, 1, 17], :]
            FstPoint, SndPoint, TrdPoint = elbow, core_of_hand , center_of_hand

            def calculate_angle_vectorized(p1, p2, p3):
                v1 = p1 - p2
                v2 = p3 - p2
                norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
                norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
                v1_normed = np.divide(v1, norm_v1, out=np.zeros_like(v1), where=norm_v1!=0)
                v2_normed = np.divide(v2, norm_v2, out=np.zeros_like(v2), where=norm_v2!=0)
                dot_product = np.clip(np.sum(v1_normed * v2_normed, axis=1), -1.0, 1.0)
                angle_rad = np.arccos(dot_product)
                return np.degrees(angle_rad)

            angles = calculate_angle_vectorized(FstPoint, SndPoint, TrdPoint)
            
            list_metrics_min_angle = []
            list_metrics_max_angle = []
            min_indices = []
            for start, end in intervals:
                start_clamped = max(0, start)
                end_clamped = min(len(angles), end)
                if start_clamped < end_clamped:
                    interval_data = angles[start_clamped:end_clamped]
                    if len(interval_data) > 0:
                        min_val = np.min(interval_data)
                        max_val = np.max(interval_data)
                        max_idx_local = np.argmax(interval_data)
                        min_idx_local = np.argmin(interval_data)
                        list_metrics_min_angle.append(min_val)
                        list_metrics_max_angle.append(max_val)
                        min_indices.append(start_clamped + min_idx_local)
                        min_indices.append(start_clamped + max_idx_local)
                    else: list_metrics_min_angle.append(np.nan)
                else: list_metrics_min_angle.append(np.nan)
            valid_metrics_min= [m for m in list_metrics_min_angle if not np.isnan(m)]
            valid_metrics_max = [m for m in list_metrics_max_angle if not np.isnan(m)]
            mean_min_dist = np.mean(valid_metrics_min) if valid_metrics_min else np.nan
            print(f"Метрика 'Угол учевого отведения': {mean_min_dist:.2f}")
            mean_max_dist = np.mean(valid_metrics_max) if valid_metrics_max else np.nan
            print(f"Метрика 'Угол локтевого отведения': {mean_max_dist:.2f}")
        
            
            
            
            
            
            angles_third_derivative = np.zeros_like(angles)
            h = dt
            if len(angles) >= 5:
                term1, term2, term3, term4 = angles[4:], -2*angles[3:-1], 2*angles[1:-3], -angles[:-4]
                angles_third_derivative[2:-2] = (term1 + term2 + term3 + term4) / (2 * h**3)
                valid_derivative = angles_third_derivative[2:-2]
                metric_smoothness = np.sqrt(np.mean(valid_derivative**2)) if len(valid_derivative) > 0 else np.nan
                print(f"Метрика 'Коэф плавности': {metric_smoothness:.2f}")
            else:
                print("Предупреждение: Недостаточно данных после start_frame_angle_analysis для расчета 3-й производной угла.")
                metric_smoothness = np.nan
                angles_third_derivative = np.array([])
        except IndexError:
             print(f"Ошибка: Недостаточно кадров ({len(keypoints_data_filtered)}) после start_frame_angle_analysis ({start_frame_angle_analysis}) для выбора точек угла.")
    else: print("Ошибка: Правая рука не найдена или недостаточно кадров для расчета метрики плавности.")

    # --- НАСТРОЙКА РАЗМЕРОВ ШРИФТОВ (без изменений) ---
    print("Настройка увеличенных размеров шрифтов для графика...")
    font_scale_factor = 2.5
    try: default_font_size = float(plt.rcParams['font.size'])
    except ValueError:
        print("Предупреждение: plt.rcParams['font.size'] не число. Используется 10.")
        default_font_size = 10.0
    new_font_size = default_font_size * font_scale_factor
    new_axes_title_size, new_axes_label_size = new_font_size*1.1, new_font_size
    new_tick_label_size, new_legend_font_size = new_font_size*0.9, new_font_size*0.9
    new_figure_title_size = new_font_size*1.2
    plt.rcParams.update({
        'font.size': new_font_size, 'axes.titlesize': new_axes_title_size,
        'axes.labelsize': new_axes_label_size/1.2, 'xtick.labelsize': new_tick_label_size,
        'ytick.labelsize': new_tick_label_size, 'legend.fontsize': new_legend_font_size/2.2,
        'figure.titlesize': new_figure_title_size
    })
    # -----------------------------------


    # --- Создание сводного графика ---
    print("Создание сводного графика метрик...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    fig.suptitle('Анализ метрик движения правой руки')

    # --- Общие параметры для графиков ---
    plot_start_frame = 0
    plot_end_frame = num_frames
    # Конвертируем в АБСОЛЮТНОЕ ВРЕМЯ для расчетов и установки пределов
    plot_start_time = plot_start_frame * dt
    plot_end_time = plot_end_frame * dt

    highlight_color = 'gainsboro'
    highlight_alpha = 0.5
    time_angle = np.arange(start_frame_angle_analysis, start_frame_angle_analysis + len(angles)) * dt
    # 1. График: Расстояние
    ax1 = axes[0]
    time_dist = np.arange(len(angles)) * dt
    if len(time_dist) > 0:
        ax1.plot(time_angle, angles, label='Угол между рукой и кистью (°)', color='tab:blue')
        min_times = [idx * dt for idx in min_indices]
        valid_min_times_indices = [(t, i) for t, i in zip(min_times, min_indices) if plot_start_time <= t < plot_end_time]
        if valid_min_times_indices:
            times_to_plot = [time_angle[int(t/dt)] for t, i in valid_min_times_indices]
            # print(times_to_plot)
            values_to_plot = [angles[i] for t, i in valid_min_times_indices]
            ax1.scatter(times_to_plot, values_to_plot, color='red', s=50, zorder=5, label='Мин. и Макс. в интервале')
    ax1.set_ylabel('Угол (°)')
    ax1.set_title('Угол отведения')
    
    ax1 = axes[0]
    # ... (весь код построения графика ax1) ...
    ax1.legend(loc='upper right') # Отображаем легенду

    # !!! ДОБАВЛЯЕМ ЭТУ СТРОКУ !!!
    # ax1.set_ylim(bottom=0)
    ax1.grid(True)
    for start, end in intervals:
        start_time, end_time = start * dt, end * dt
        display_start = max(start_time, plot_start_time)
        display_end = min(end_time, plot_end_time)
        if display_start < display_end:
             ax1.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')
    ax1.legend(loc='upper right')

    # 2. График: Дрожание
    ax2 = axes[1]
    time_delta = np.arange(start_frame_analysis, start_frame_analysis + len(delta_x)) * dt
    if len(time_delta) == len(delta_x) > 0:
        ax2.plot(time_delta, delta_x, label='X запястья', alpha=0.7)
        ax2.plot(time_delta, delta_y, label='Y запястья', alpha=0.7)
        ax2.plot(time_delta, delta_z, label='Z запястья', alpha=0.7)
    ax2.set_ylabel('Координата (мм)')
    ax2.set_title('Дрожание запястья (Координаты)')
    ax2.grid(True)
    for start, end in intervals:
        start_time, end_time = start * dt, end * dt
        display_start = max(start_time, plot_start_time)
        display_end = min(end_time, plot_end_time)
        if display_start < display_end:
            ax2.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')
    ax2.legend(loc='upper right')

    # 3. График: Плавность
    ax3 = axes[2]
    # time_angle = np.arange(start_frame_angle_analysis, start_frame_angle_analysis + len(angles)) * dt
    valid_deriv_start_frame = start_frame_angle_analysis + 2
    valid_deriv_end_frame = start_frame_angle_analysis + len(angles) - 2
    valid_derivative_data = angles_third_derivative[2:-2]
    time_deriv_valid = np.array([])
    if valid_deriv_start_frame < valid_deriv_end_frame:
         time_deriv_valid = np.arange(valid_deriv_start_frame, valid_deriv_end_frame) * dt
         if len(time_deriv_valid) != len(valid_derivative_data):
              print(f"Warning: Len mismatch time({len(time_deriv_valid)}) vs deriv({len(valid_derivative_data)}).")
              min_len = min(len(time_deriv_valid), len(valid_derivative_data))
              time_deriv_valid = time_deriv_valid[:min_len]
              valid_derivative_data = valid_derivative_data[:min_len]

    line1 = None
    if len(time_angle) == len(angles) > 0:
        line1, = ax3.plot(time_angle, angles, label='Угол между рукой и кистью (°)', color='tab:blue')
        ax3.set_ylabel('Угол (°)', color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

    ax3b = ax3.twinx()
    line2 = None
    if len(time_deriv_valid) > 0 and len(valid_derivative_data) == len(time_deriv_valid):
        line2, = ax3b.plot(time_deriv_valid, valid_derivative_data, label='3-я производная угла (Рывок)', color='tab:red', alpha=0.6)
        ax3b.set_ylabel('Рывок (°/с³)', color='tab:red')
        ax3b.tick_params(axis='y', labelcolor='tab:red')
    else: ax3b.set_yticks([])

    # !!! ИЗМЕНЕНИЕ: Меняем подпись оси X !!!
    ax3.set_xlabel('Время от начала отображения (с)') # Новая подпись
    ax3.set_title('Угловая плавность движения (Угол и Рывок)')
    ax3.grid(True)
    for start, end in intervals:
         start_time, end_time = start * dt, end * dt
         display_start = max(start_time, plot_start_time)
         display_end = min(end_time, plot_end_time)
         if display_start < display_end:
            ax3.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')

    lines = []
    labels = []
    if line1: lines.append(line1); labels.append(line1.get_label())
    if line2: lines.append(line2); labels.append(line2.get_label())
    if lines: ax3.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.9))

    # --- Установка общего диапазона оси X В АБСОЛЮТНЫХ СЕКУНДАХ ---
    # Пределы остаются в абсолютном времени, чтобы данные отображались корректно
    ax3.set_xlim(plot_start_time, plot_end_time)

    # !!! ИЗМЕНЕНИЕ: Добавляем FuncFormatter для оси X !!!
    # Эта функция будет вызвана для каждого тика на оси X.
    # 'value' - это абсолютное время (например, 2.5с)
    # Мы возвращаем строку, представляющую время относительно начала отображения.
    # Формат ':.1f' означает одно число после запятой.
    def format_time_ticks(value, tick_position):
        relative_time = value - plot_start_time
        # Можно добавить маленькое значение, чтобы избежать -0.0 из-за точности float
        if abs(relative_time) < 1e-9:
             relative_time = 0.0
        return f"{relative_time:.1f}" # Форматируем с 1 знаком после запятой

    # Применяем форматтер к основной оси X (ax3). Т.к. sharex=True, он применится ко всем.
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(format_time_ticks))
    # --------------------------------------------------

    # --- Финальная компоновка и отображение ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_filename = 'analysis_plot_relative_time_axis.svg' # Новое имя файла
    # fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    # print(f"График сохранен как {save_filename}")
    plt.show()

    print("Завершено.") 
    
    
    