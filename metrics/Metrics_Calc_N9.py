import numpy as np
import matplotlib.pyplot as plt # Импортируем matplotlib
from matplotlib import ticker # !!! ДОБАВЛЯЕМ ИМПОРТ TICKER !!!
# from mpl_toolkits.mplot3d import Axes3D # Этот импорт больше не нужен
import math
from scipy.signal import butter, lfilter
import os # Добавим для проверки файлов



import math


import numpy as np

def calculate_angle_vectorized(p1, p2, p3, p4, reference_axis=np.array([0.0, 0.0, 1.0])):
    """
    Вычисляет знаковый угол (в диапазоне [-180, 180]) между векторами
    v1 = p1 - p2 и v2 = p3 - p4 относительно опорной оси reference_axis.
    (Версия с исправленным broadcasting'ом)
    # ... (остальное описание функции без изменений) ...
    """
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p4)

    # --- Вычисление модуля угла (0 до 180) ---
    norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True) # Shape (N, 1)
    norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True) # Shape (N, 1)

    # Избегаем деления на ноль для нулевых векторов
    valid_norms = (norm_v1 != 0) & (norm_v2 != 0) # Shape (N, 1)
    # Инициализируем результат как nan там, где нормы нулевые
    signed_angle_deg = np.full(v1.shape[0], np.nan)

    # Индексируем с помощью .flatten(), чтобы получить 1D булев массив для выбора строк
    # Это проще для индексации и v1, и v2, и reference_axis (если она N, 3)
    valid_indices_flat = valid_norms.flatten() # Shape (N,)

    # Работаем только с валидными векторами
    v1_valid = v1[valid_indices_flat] # Shape (num_valid, 3)
    v2_valid = v2[valid_indices_flat] # Shape (num_valid, 3)
    # Отфильтрованные нормы будут (num_valid,) после такого индексирования
    norm_v1_valid = norm_v1[valid_indices_flat, 0] # Shape (num_valid,)
    norm_v2_valid = norm_v2[valid_indices_flat, 0] # Shape (num_valid,)

    if v1_valid.shape[0] == 0: # Если нет валидных пар
        return signed_angle_deg

    # ----- ИСПРАВЛЕНИЕ ЗДЕСЬ -----
    # Добавляем ось для правильного broadcasting'а: (num_valid,) -> (num_valid, 1)
    v1_normed = v1_valid / norm_v1_valid[:, np.newaxis] # (num_valid, 3) / (num_valid, 1)
    v2_normed = v2_valid / norm_v2_valid[:, np.newaxis] # (num_valid, 3) / (num_valid, 1)
    # ------------------------------

    # Скалярное произведение (косинус угла)
    dot_product = np.clip(np.sum(v1_normed * v2_normed, axis=1), -1.0, 1.0) # Shape (num_valid,)
    angle_rad_unsigned = np.arccos(dot_product) # Угол от 0 до pi, shape (num_valid,)

    # --- Определение знака угла ---
    cross_prod = np.cross(v1_normed, v2_normed, axisa=1, axisb=1) # Shape (num_valid, 3)

    # Нормализуем опорную ось (или оси), если нужно
    reference_axis = np.asarray(reference_axis)
    if reference_axis.ndim == 1:
        reference_axis = reference_axis.reshape(1, 3) # Делаем (1, 3) для broadcasting
    norm_ref_axis = np.linalg.norm(reference_axis, axis=1, keepdims=True) # Shape (M, 1) где M=1 или M=N
    # Избегаем деления на ноль для reference_axis
    valid_ref_axis = (norm_ref_axis != 0) # Shape (M, 1)
    ref_axis_normed = np.divide(reference_axis, norm_ref_axis,
                                out=np.zeros_like(reference_axis),
                                where=valid_ref_axis) # Shape (M, 3)

    # Выбираем соответствующие опорные оси для валидных пар
    if ref_axis_normed.shape[0] == v1.shape[0]: # Если для каждой пары своя ось (N, 3)
         ref_axis_normed_valid = ref_axis_normed[valid_indices_flat] # Shape (num_valid, 3)
         zero_ref_axis_mask_valid = ~valid_ref_axis.flatten()[valid_indices_flat] # Shape (num_valid,)
    elif ref_axis_normed.shape[0] == 1: # Если одна ось для всех (1, 3)
         ref_axis_normed_valid = ref_axis_normed # Shape (1, 3), broadcasting сработает с (num_valid, 3)
         zero_ref_axis_mask_valid = np.full(v1_valid.shape[0], fill_value=not valid_ref_axis.item()) # Shape (num_valid,)
    else:
         raise ValueError("reference_axis должен быть формы (3,), (1, 3) или (N, 3)")

    # Знак определяется знаком скалярного произведения
    sign_dot = np.sum(cross_prod * ref_axis_normed_valid, axis=1) # Shape (num_valid,)

    # Присваиваем знак. Используем np.sign, но обрабатываем 0 отдельно, если нужно
    # (хотя для 0 и 180 знак не так важен, можно оставить sign=1)
    signs = np.where(sign_dot >= 0, 1.0, -1.0) # Shape (num_valid,)

    # Применяем знак к углу
    signed_angle_rad = angle_rad_unsigned * signs # Shape (num_valid,)

    # Преобразуем в градусы
    signed_angle_deg_valid = np.degrees(signed_angle_rad) # Shape (num_valid,)

    # Устанавливаем nan там, где опорная ось была нулевой
    signed_angle_deg_valid[zero_ref_axis_mask_valid] = np.nan

    # Помещаем вычисленные значения обратно в исходный массив
    # Используем ту же плоскую маску для присвоения
    signed_angle_deg[valid_indices_flat] = signed_angle_deg_valid

    return signed_angle_deg

# --- Пример использования (остается тем же) ---
# ... (код примера без изменений) ...

def distance(p1, p2):
  """Вычисляет евклидово расстояние между двумя точками (кортежами/списками)."""
  return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def polygon_perimeter(vertices):
  """Вычисляет периметр многоугольника, заданного списком вершин."""
  perimeter = 0.0
  num_vertices = len(vertices)
  if num_vertices < 3:
    return 0.0 # Не многоугольник

  for i in range(num_vertices):
    p1 = vertices[i]
    # Берем следующую вершину, замыкая круг для последней точки
    p2 = vertices[(i + 1) % num_vertices]
    perimeter += distance(p1, p2)
  return perimeter

def polygon_area(vertices):
  """Вычисляет площадь многоугольника по формуле Гаусса (метод шнуровки)."""
  area = 0.0
  num_vertices = len(vertices)
  if num_vertices < 3:
    return 0.0 # Не многоугольник

  for i in range(num_vertices):
    j = (i + 1) % num_vertices # Индекс следующей вершины (с замыканием)
    x1, y1 = vertices[i]
    x2, y2 = vertices[j]
    area += (x1 * y2)
    area -= (y1 * x2)
  area = 0.5 * abs(area) # Площадь всегда положительна
  return area

def isoperimetric_ratio(vertices):
  """Вычисляет изопериметрическое соотношение для многоугольника."""
  if len(vertices) < 3:
      print("Ошибка: Для расчета нужно минимум 3 точки.")
      return None, None, None

  area = polygon_area(vertices)
  perimeter = polygon_perimeter(vertices)

  if perimeter == 0:
    print("Ошибка: Периметр равен нулю. Невозможно рассчитать соотношение.")
    # Это может случиться, если все точки совпадают или лежат на одной прямой (площадь тоже будет 0)
    return area, perimeter, None # Или вернуть 0, или None

  # Формула изопериметрического соотношения
  ratio = (4 * math.pi * area) / (perimeter**2)
  return area, perimeter, ratio



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
        (0, 17)
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
    num_frames = 800
    start_frame_analysis = 70 # С какого кадра начинать анализ для Delta
    start_frame_angle_analysis = 60 # С какого кадра начинать анализ для углов
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
         8, 6, 6, 6, 5      # Pinky
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
             keypoints_data.append([np.zeros((22, 3)), np.zeros((22, 3))])
             # keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])
             missing_files += 1
             continue

        keypoints_raw = read_hand_skeleton_data(file_path, thickness_array)
        # print(keypoints_raw)
        keypoints_data.append(keypoints_raw)

        # if os.path.exists(file_path_aligned):
        #      keypoints_raw_aligned = read_hand_skeleton_data(file_path_aligned, thickness_array)
        #      keypoints_data_aligned.append(keypoints_raw_aligned)
        # else:
        #      keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])

    if missing_files > 0: print(f"Предупреждение: Пропущено {missing_files} кадров.")
    # print(keypoints_data)
    keypoints_data = np.array(keypoints_data)
    # print(keypoints_data)
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
    # print(keypoints_data)
    right_hand_idx = 0
    intervals = [[50, 100], [110, 160], [180, 230], [240, 300], [310, 360],[370,420],[430,480],[490,540]]
    # it_bf=[[50,100],[430,480],[490,540]]
    it_bf=[[60,120],[140,190],[210,260],[270,330]]
    it_mi=[[110,160],[370,420]]
    it_ma=[[180,230],[310,360]]
    it_miz=[[240,300]]
    
    
    df1=[np.linalg.norm(i) for i in (keypoints_data[:, 0, 4, :2]-keypoints_data[:, 0, 8 , :2])]
    df2=[np.linalg.norm(i) for i in (keypoints_data[:, 0, 4, :2]-keypoints_data[:, 0, 12, :2])]
    df3=[np.linalg.norm(i) for i in (keypoints_data[:, 0, 4, :2]-keypoints_data[:, 0, 16, :2])]
    df4=[np.linalg.norm(i) for i in (keypoints_data[:, 0, 4, :2]-keypoints_data[:, 0, 20, :2])]
    # print(df1)
    # plt.plot(df1)
    # plt.plot(df2)
    # plt.plot(df3)
    # plt.plot(df4)
    # plt.show()
    
    
    P1=keypoints_data[:, 0, 2, :3]
    P2=keypoints_data[:, 0, 3, :3]
    P3=keypoints_data[:, 0, 4, :3]
    P4=keypoints_data[:, 0, 5, :3]
    P5=keypoints_data[:, 0, 6, :3]
    P6=keypoints_data[:, 0, 7, :3]
    P7=keypoints_data[:, 0, 8, :3]    
    # iso=[]
    # for i in range(len(keypoints_data)):
    #     calculated_area, calculated_perimeter, calculated_ratio = isoperimetric_ratio([P1[i],P2[i],P3[i],P4[i],P5[i],P6[i],P7[i]])
    #     iso.append(np.sqrt(calculated_ratio))
    # plt.plot(iso)
    # plt.show() 
    sa=calculate_angle_vectorized(P3,P2,P7,P6,  np.array([0.0, 0.0, 1.0]))
    m1=[]
    for pr in it_bf:
        m1.append(max(sa[pr[0]:pr[1]]))
    m1_res=np.mean(m1)
    mxa=[]
    for i,s in enumerate(sa):
        if i>3:
            if abs(sa[i]-sa[i-1])>80:
                sa[i]=sa[i-1]
    for start, end in it_bf:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        # print(display_start ,display_end)
        mx=max((sa[display_start:display_end]))
        mxa.append(mx)
        if display_start < display_end:
            plt.axvspan(display_start/30, display_end/30, color='gainsboro', alpha=0.5, label='_nolegend_')

    print('-----------------')
    print(mxa)
    print(np.mean(mxa))
    # set x axil divided by 30
    
    plt.title('Угол между крайней фалангой указательного и большого пальцев (°)')
    plt.xlim(0,23.5)
    # sa=calculate_angle_vectorized(P3,P2,P7,P6, np.array([0.0, 0.0, 1.0]))
    plt.plot(np.arange(len(sa))/30,sa)
    plt.show()      
    
    P1=keypoints_data[:, 0, 2, :3]
    P2=keypoints_data[:, 0, 3, :3]
    P3=keypoints_data[:, 0, 4, :3]
    P4=keypoints_data[:, 0, 9, :3]
    P5=keypoints_data[:, 0, 10, :3]
    P6=keypoints_data[:, 0, 11, :3]
    P7=keypoints_data[:, 0, 12, :3]    
    iso=[]
    # for i in range(len(keypoints_data)):
    #     calculated_area, calculated_perimeter, calculated_ratio = isoperimetric_ratio([P1[i],P2[i],P3[i],P4[i],P5[i],P6[i],P7[i]])
    #     iso.append(np.sqrt(calculated_ratio))
    # plt.plot(iso)
    # plt.show()     
    
    sa=calculate_angle_vectorized(P3,P2,P7,P6,  np.array([1.0, 1.0, 1.0]))
    nan_mask = np.isnan(sa)
    sa[nan_mask]=0
    m1=[]
    
    for pr in it_mi:
        m1.append(max(sa[pr[0]:pr[1]]))
        # print(sa[pr[0]:pr[1]])
    m1_res=np.mean(m1)
    mxa=[]
    for start, end in it_mi:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        # print(display_start ,display_end)
        mx=max((sa[display_start:display_end]))
        mxa.append(mx)
        if display_start < display_end:
            plt.axvspan(display_start/30 , display_end/30, color='gainsboro', alpha=0.5, label='_nolegend_')
    # for i,s in enumerate(sa):
    #     if i>3:
    #         if abs(sa[i]-sa[i-1])>70:
    #             sa[i]=sa[i-1]
    print(np.mean(mxa))
    # sa=calculate_angle_vectorized(P3,P2,P7,P6, np.array([0.0, 0.0, 1.0]))
    plt.plot((np.arange(len(sa)))/30, sa)
    plt.title('Угол между крайней фалангой среднего и большого пальцев (°)')
    # plt.xlim(0, 10)
    plt.show()   
    
    
    P1=keypoints_data[:, 0, 2, :3]
    P2=keypoints_data[:, 0, 3, :3]
    P3=keypoints_data[:, 0, 4, :3]
    P4=keypoints_data[:, 0, 13, :3]
    P5=keypoints_data[:, 0, 14, :3]
    P6=keypoints_data[:, 0, 15, :3]
    P7=keypoints_data[:, 0, 16, :3]    
    iso=[]
    # for i in range(len(keypoints_data)):
    #     calculated_area, calculated_perimeter, calculated_ratio = isoperimetric_ratio([P1[i],P2[i],P3[i],P4[i],P5[i],P6[i],P7[i]])
    #     iso.append(np.sqrt(calculated_ratio))
    # # plt.plot(iso)
    # plt.show()   
    
    sa=calculate_angle_vectorized(P3,P2,P7,P6,  np.array([0, .0, 1.0]))
    # print(sa)
    nan_mask = np.isnan(sa)
    sa[nan_mask]=0
    m1=[]
    print(len(sa))
    for pr in it_bf:
        m1.append(max(sa[pr[0]:pr[1]]))
        print(sa[pr[0]:pr[1]])
    m1_res=np.mean(m1)
    mxa=[]
    for start, end in it_ma:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        # print(display_start ,display_end)
        mx=max((sa[display_start:display_end]))
        mxa.append(mx)
        if display_start < display_end:
            plt.axvspan(display_start/30 , display_end/30 , color='gainsboro', alpha=0.5, label='_nolegend_')
    for i,s in enumerate(sa):
        if i>3:
            if (sa[i]-sa[i-1])>120:
                sa[i]=sa[i-1]
            # if abs(sa[i]-sa[i-2])>75:
            #     sa[i]=sa[i-1]
    print(np.mean(mxa))
    # sa=calculate_angle_vectorized(P3,P2,P7,P6, np.array([0.0, 0.0, 1.0]))
    plt.plot((np.arange(len(sa)))/30, sa)
    plt.title('Угол между крайней фалангой безымянного и большого пальцев (°)')
    # plt.xlim(0, 5.9)
    plt.show()   
    
    
    
    P1=keypoints_data[:, 0, 2, :3]
    P2=keypoints_data[:, 0, 3, :3]
    P3=keypoints_data[:, 0, 4, :3]
    P4=keypoints_data[:, 0, 17, :3]
    P5=keypoints_data[:, 0, 18, :3]
    P6=keypoints_data[:, 0, 19, :3]
    P7=keypoints_data[:, 0, 20, :3]    
    iso=[]
    # for i in range(len(keypoints_data)):
    #     calculated_area, calculated_perimeter, calculated_ratio = isoperimetric_ratio([P1[i],P2[i],P3[i],P4[i],P5[i],P6[i],P7[i]])
    #     iso.append(np.sqrt(calculated_ratio))
    # plt.plot(iso)
    # plt.show()   
    
    sa=calculate_angle_vectorized(P3,P2,P7,P6,  np.array([0.0, 0.0, 1.0]))
    m1=[]
    nan_mask = np.isnan(sa)
    sa[nan_mask]=0
    for pr in it_bf:
        m1.append(max(sa[pr[0]:pr[1]]))
    m1_res=np.mean(m1)
    mxa=[]
    for start, end in it_miz:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        # print(display_start ,display_end)
        mx=max((sa[display_start:display_end]))
        mxa.append(mx)
        if display_start < display_end:
            plt.axvspan(display_start/30 -27, display_end/30 -27, color='gainsboro', alpha=0.5, label='_nolegend_')
    # for i,s in enumerate(sa):
    #     if i>3:
    #         if abs(sa[i]-sa[i-1])>80:
    #             sa[i]=sa[i-1]
    
    print(np.mean(mxa))
    # sa=calculate_angle_vectorized(P3,P2,P7,P6, np.array([0.0, 0.0, 1.0]))
    plt.plot((np.arange(len(sa)))/30,sa)
    # plt.xlim(0, 9.9)
    plt.title('Угол между крайней фалангой мизинца и большого пальца (°)')
    plt.show()   
    
 
    
    
       
    # y_dist_diff = np.array([])
    # min_indices = []
    # mean_min_dist = np.nan
    # delta_x, delta_y, delta_z = np.array([]), np.array([]), np.array([])
    # metric_delta = np.nan
    # angles = np.array([])
    # angles_third_derivative = np.array([])
    # metric_smoothness = np.nan
    # keypoints_data_filtered=keypoints_data
    # print(keypoints_data_filtered)
    # # 1. Метрика: Мин расстояние
    # if keypoints_data_filtered.shape[1] > right_hand_idx and keypoints_data_filtered.shape[0] > 0:
    #     z_thumb_tip = keypoints_data_filtered[:, right_hand_idx, 4, 2] 
    #     z_pinky_tip = keypoints_data_filtered[:, right_hand_idx, 17, 2]
    #     y_dist_diff = z_thumb_tip - z_pinky_tip
    #     list_metrics_dist = []
    #     min_indices = []
    #     for start, end in intervals:
    #         start_clamped = max(0, start)
    #         end_clamped = min(len(y_dist_diff), end)
    #         if start_clamped < end_clamped:
    #             interval_data = y_dist_diff[start_clamped:end_clamped]
    #             if len(interval_data) > 0:
    #                 min_val = np.min(interval_data)
    #                 min_idx_local = np.argmin(interval_data)
    #                 list_metrics_dist.append(min_val)
    #                 min_indices.append(start_clamped + min_idx_local)
    #             else: list_metrics_dist.append(np.nan)
    #         else: list_metrics_dist.append(np.nan)
    #     valid_metrics_dist = [m for m in list_metrics_dist if not np.isnan(m)]
    #     mean_min_dist = np.mean(valid_metrics_dist) if valid_metrics_dist else np.nan
    #     print(f"Метрика 'Мин расстояние до б пальца': {mean_min_dist:.2f}")
    # else: print("Ошибка: Правая рука не найдена или недостаточно данных для расчета метрики расстояния.")

    # # 2. Метрика: Delta (Дрожание)
    # if keypoints_data_filtered.shape[1] > right_hand_idx and len(keypoints_data_filtered) > start_frame_analysis:
    #     wrist_coords = keypoints_data_filtered[start_frame_analysis:, right_hand_idx, 0, :]
    #     delta_z, delta_y, delta_x = wrist_coords[:, 2], wrist_coords[:, 1], wrist_coords[:, 0]
    #     if len(delta_x) > 1:
    #         std_z, std_y, std_x = np.std(delta_z), np.std(delta_y), np.std(delta_x)
    #         metric_delta = np.sqrt(std_z**2 + std_y**2 + std_x**2)
    #         print(f"Метрика 'Delta (Дрожание)': {metric_delta:.2f}")
    #     else:
    #         print("Предупреждение: Недостаточно данных после start_frame_analysis для расчета Delta.")
    #         metric_delta = np.nan
    #         delta_x, delta_y, delta_z = np.array([]), np.array([]), np.array([])
    # else: print("Ошибка: Правая рука не найдена или недостаточно кадров для расчета метрики Delta.")

    # # 3. Метрика: Коэф плавности (Jerk)
    # if keypoints_data_filtered.shape[1] > right_hand_idx and len(keypoints_data_filtered) > start_frame_angle_analysis:
    #     try:
    #         points_for_angle = keypoints_data_filtered[start_frame_angle_analysis:, right_hand_idx, [4, 1, 17], :]
    #         FstPoint, SndPoint, TrdPoint = points_for_angle[:, 0, :], points_for_angle[:, 1, :], points_for_angle[:, 2, :]

    #         def calculate_angle_vectorized(p1, p2, p3):
    #             v1 = p2 - p1
    #             v2 = p3 - p2
    #             norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
    #             norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
    #             v1_normed = np.divide(v1, norm_v1, out=np.zeros_like(v1), where=norm_v1!=0)
    #             v2_normed = np.divide(v2, norm_v2, out=np.zeros_like(v2), where=norm_v2!=0)
    #             dot_product = np.clip(np.sum(v1_normed * v2_normed, axis=1), -1.0, 1.0)
    #             angle_rad = np.arccos(dot_product)
    #             return np.degrees(angle_rad)

    #         angles = calculate_angle_vectorized(FstPoint, SndPoint, TrdPoint)
    #         angles_third_derivative = np.zeros_like(angles)
    #         h = dt
    #         if len(angles) >= 5:
    #             term1, term2, term3, term4 = angles[4:], -2*angles[3:-1], 2*angles[1:-3], -angles[:-4]
    #             angles_third_derivative[2:-2] = (term1 + term2 + term3 + term4) / (2 * h**3)
    #             valid_derivative = angles_third_derivative[2:-2]
    #             metric_smoothness = np.sqrt(np.mean(valid_derivative**2)) if len(valid_derivative) > 0 else np.nan
    #             print(f"Метрика 'Коэф плавности': {metric_smoothness:.2f}")
    #         else:
    #             print("Предупреждение: Недостаточно данных после start_frame_angle_analysis для расчета 3-й производной угла.")
    #             metric_smoothness = np.nan
    #             angles_third_derivative = np.array([])
    #     except IndexError:
    #          print(f"Ошибка: Недостаточно кадров ({len(keypoints_data_filtered)}) после start_frame_angle_analysis ({start_frame_angle_analysis}) для выбора точек угла.")
    # else: print("Ошибка: Правая рука не найдена или недостаточно кадров для расчета метрики плавности.")

    # # --- НАСТРОЙКА РАЗМЕРОВ ШРИФТОВ (без изменений) ---
    # print("Настройка увеличенных размеров шрифтов для графика...")
    # font_scale_factor = 2.5
    # try: default_font_size = float(plt.rcParams['font.size'])
    # except ValueError:
    #     print("Предупреждение: plt.rcParams['font.size'] не число. Используется 10.")
    #     default_font_size = 10.0
    # new_font_size = default_font_size * font_scale_factor
    # new_axes_title_size, new_axes_label_size = new_font_size*1.1, new_font_size
    # new_tick_label_size, new_legend_font_size = new_font_size*0.9, new_font_size*0.9
    # new_figure_title_size = new_font_size*1.2
    # plt.rcParams.update({
    #     'font.size': new_font_size, 'axes.titlesize': new_axes_title_size,
    #     'axes.labelsize': new_axes_label_size/1.2, 'xtick.labelsize': new_tick_label_size,
    #     'ytick.labelsize': new_tick_label_size, 'legend.fontsize': new_legend_font_size/2.2,
    #     'figure.titlesize': new_figure_title_size
    # })
    # # -----------------------------------


    # # --- Создание сводного графика ---
    # print("Создание сводного графика метрик...")
    # fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    # fig.suptitle('Анализ метрик движения правой руки')

    # # --- Общие параметры для графиков ---
    # plot_start_frame = 60
    # plot_end_frame = num_frames
    # # Конвертируем в АБСОЛЮТНОЕ ВРЕМЯ для расчетов и установки пределов
    # plot_start_time = plot_start_frame * dt
    # plot_end_time = plot_end_frame * dt

    # highlight_color = 'gainsboro'
    # highlight_alpha = 0.5

    # # 1. График: Расстояние
    # ax1 = axes[0]
    # time_dist = np.arange(len(y_dist_diff)) * dt
    # if len(time_dist) > 0:
    #     ax1.plot(time_dist, y_dist_diff, label='Разница Z (Большой - Мизинец)')
    #     min_times = [idx * dt for idx in min_indices]
    #     valid_min_times_indices = [(t, i) for t, i in zip(min_times, min_indices) if plot_start_time <= t < plot_end_time]
    #     if valid_min_times_indices:
    #         times_to_plot = [t for t, i in valid_min_times_indices]
    #         values_to_plot = [y_dist_diff[i] for t, i in valid_min_times_indices]
    #         ax1.scatter(times_to_plot, values_to_plot, color='red', s=50, zorder=5, label='Мин. в интервале')
    # ax1.set_ylabel('Разница Z (мм)')
    # ax1.set_title('Расстояние м/у большим пальцем и мизинцем')
    
    # ax1 = axes[0]
    # # ... (весь код построения графика ax1) ...
    # ax1.legend(loc='upper right') # Отображаем легенду

    # # !!! ДОБАВЛЯЕМ ЭТУ СТРОКУ !!!
    # # ax1.set_ylim(bottom=0)
    # ax1.grid(True)
    # for start, end in intervals:
    #     start_time, end_time = start * dt, end * dt
    #     display_start = max(start_time, plot_start_time)
    #     display_end = min(end_time, plot_end_time)
    #     if display_start < display_end:
    #          ax1.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')
    # ax1.legend(loc='upper right')

    # # 2. График: Дрожание
    # ax2 = axes[1]
    # time_delta = np.arange(start_frame_analysis, start_frame_analysis + len(delta_x)) * dt
    # if len(time_delta) == len(delta_x) > 0:
    #     ax2.plot(time_delta, delta_x, label='X запястья', alpha=0.7)
    #     ax2.plot(time_delta, delta_y, label='Y запястья', alpha=0.7)
    #     ax2.plot(time_delta, delta_z, label='Z запястья', alpha=0.7)
    # ax2.set_ylabel('Координата (мм)')
    # ax2.set_title('Дрожание запястья (Координаты)')
    # ax2.grid(True)
    # for start, end in intervals:
    #     start_time, end_time = start * dt, end * dt
    #     display_start = max(start_time, plot_start_time)
    #     display_end = min(end_time, plot_end_time)
    #     if display_start < display_end:
    #         ax2.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')
    # ax2.legend(loc='upper right')

    # # 3. График: Плавность
    # ax3 = axes[2]
    # time_angle = np.arange(start_frame_angle_analysis, start_frame_angle_analysis + len(angles)) * dt
    # valid_deriv_start_frame = start_frame_angle_analysis + 2
    # valid_deriv_end_frame = start_frame_angle_analysis + len(angles) - 2
    # valid_derivative_data = angles_third_derivative[2:-2]
    # time_deriv_valid = np.array([])
    # if valid_deriv_start_frame < valid_deriv_end_frame:
    #      time_deriv_valid = np.arange(valid_deriv_start_frame, valid_deriv_end_frame) * dt
    #      if len(time_deriv_valid) != len(valid_derivative_data):
    #           print(f"Warning: Len mismatch time({len(time_deriv_valid)}) vs deriv({len(valid_derivative_data)}).")
    #           min_len = min(len(time_deriv_valid), len(valid_derivative_data))
    #           time_deriv_valid = time_deriv_valid[:min_len]
    #           valid_derivative_data = valid_derivative_data[:min_len]

    # line1 = None
    # if len(time_angle) == len(angles) > 0:
    #     line1, = ax3.plot(time_angle, angles, label='Угол между пальцами (°)', color='tab:blue')
    #     ax3.set_ylabel('Угол (°)', color='tab:blue')
    #     ax3.tick_params(axis='y', labelcolor='tab:blue')

    # ax3b = ax3.twinx()
    # line2 = None
    # if len(time_deriv_valid) > 0 and len(valid_derivative_data) == len(time_deriv_valid):
    #     line2, = ax3b.plot(time_deriv_valid, valid_derivative_data, label='3-я производная угла (Рывок)', color='tab:red', alpha=0.6)
    #     ax3b.set_ylabel('Рывок (°/с³)', color='tab:red')
    #     ax3b.tick_params(axis='y', labelcolor='tab:red')
    # else: ax3b.set_yticks([])

    # # !!! ИЗМЕНЕНИЕ: Меняем подпись оси X !!!
    # ax3.set_xlabel('Время от начала отображения (с)') # Новая подпись
    # ax3.set_title('Угловая плавность движения (Угол и Рывок)')
    # ax3.grid(True)
    # for start, end in intervals:
    #      start_time, end_time = start * dt, end * dt
    #      display_start = max(start_time, plot_start_time)
    #      display_end = min(end_time, plot_end_time)
    #      if display_start < display_end:
    #         ax3.axvspan(display_start, display_end, color=highlight_color, alpha=highlight_alpha, label='_nolegend_')

    # lines = []
    # labels = []
    # if line1: lines.append(line1); labels.append(line1.get_label())
    # if line2: lines.append(line2); labels.append(line2.get_label())
    # if lines: ax3.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.9))

    # # --- Установка общего диапазона оси X В АБСОЛЮТНЫХ СЕКУНДАХ ---
    # # Пределы остаются в абсолютном времени, чтобы данные отображались корректно
    # ax3.set_xlim(plot_start_time, plot_end_time)

    # # !!! ИЗМЕНЕНИЕ: Добавляем FuncFormatter для оси X !!!
    # # Эта функция будет вызвана для каждого тика на оси X.
    # # 'value' - это абсолютное время (например, 2.5с)
    # # Мы возвращаем строку, представляющую время относительно начала отображения.
    # # Формат ':.1f' означает одно число после запятой.
    # def format_time_ticks(value, tick_position):
    #     relative_time = value - plot_start_time
    #     # Можно добавить маленькое значение, чтобы избежать -0.0 из-за точности float
    #     if abs(relative_time) < 1e-9:
    #          relative_time = 0.0
    #     return f"{relative_time:.1f}" # Форматируем с 1 знаком после запятой

    # # Применяем форматтер к основной оси X (ax3). Т.к. sharex=True, он применится ко всем.
    # ax3.xaxis.set_major_formatter(ticker.FuncFormatter(format_time_ticks))
    # # --------------------------------------------------

    # # --- Финальная компоновка и отображение ---
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # save_filename = 'analysis_plot_relative_time_axis.svg' # Новое имя файла
    # # fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    # # print(f"График сохранен как {save_filename}")
    # plt.show()

    # print("Завершено.")