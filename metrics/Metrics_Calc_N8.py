import numpy as np
import matplotlib.pyplot as plt # Импортируем matplotlib
from matplotlib import ticker # !!! ДОБАВЛЯЕМ ИМПОРТ TICKER !!!
# from mpl_toolkits.mplot3d import Axes3D # Этот импорт больше не нужен
import math
from scipy.signal import butter, lfilter
import os # Добавим для проверки файлов


ep=[[0,100],[105,230],[240,360],[370, 480], [490,600], [610,725], [730,835],[860,970],[980,1090]]

def calculate_angle_vectorized(p1, p2, p3, p4):
    v1 = p1 - p2
    v2 = p3 - p4
    norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1_normed = np.divide(v1, norm_v1, out=np.zeros_like(v1), where=norm_v1!=0)
    v2_normed = np.divide(v2, norm_v2, out=np.zeros_like(v2), where=norm_v2!=0)
    dot_product = np.clip(np.sum(v1_normed * v2_normed, axis=1), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)
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
    num_frames = 1177
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
             keypoints_data.append([np.zeros((22, 3)), np.zeros((22, 3))])
             # keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])
             missing_files += 1
             continue

        keypoints_raw = read_hand_skeleton_data(file_path, thickness_array)
        keypoints_data.append(keypoints_raw)
        # print(keypoints_raw[0].shape)
        

        # if os.path.exists(file_path_aligned):
        #      keypoints_raw_aligned = read_hand_skeleton_data(file_path_aligned, thickness_array)
        #      keypoints_data_aligned.append(keypoints_raw_aligned)
        # else:
        #      keypoints_data_aligned.append([np.zeros((21, 3)), np.zeros((21, 3))])

    if missing_files > 0: print(f"Предупреждение: Пропущено {missing_files} кадров.")

    keypoints_data = np.array(keypoints_data)
    # keypoints_data_aligned = np.array(keypoints_data_aligned)
    print(f"Данные загружены. Форма основного файла: {keypoints_data.shape}")
    print(keypoints_data[100][0])

    for i in range(keypoints_data.shape[0]):
        keypoints_data[i][0][21] =[-143.6, -221.5, -967.0]
        keypoints_data[i][1][21] =[619.6, -228.6, -960.0]
    angles_r_hand=[] 
    right_hand_idx = 0  
    elbow = keypoints_data[:, right_hand_idx, 21] 
    core_of_hand = keypoints_data[:, right_hand_idx, 0]
    center_of_hand = (keypoints_data[:, right_hand_idx, 10]+keypoints_data[:, right_hand_idx, 14])/2
    FST=elbow-core_of_hand
    SCD=center_of_hand-core_of_hand
    # print(FST)
    FstPoint, SndPoint, TrdPoint = elbow, core_of_hand , center_of_hand
    for i in range((FST.shape[0])):
        # print(np.linalg.norm(FST[i]))
        angles_r_hand.append(np.degrees(np.arcsin((FST[i] @ SCD[i])/(np.linalg.norm(FST[i])*np.linalg.norm(SCD[i])))))


    angles_l_hand=[] 
    right_hand_idx = 1  
    elbow = keypoints_data[:, right_hand_idx, 21] 
    core_of_hand = keypoints_data[:, right_hand_idx, 0]
    center_of_hand = (keypoints_data[:, right_hand_idx, 10]+keypoints_data[:, right_hand_idx, 14])/2
    
    FST=elbow-core_of_hand
    SCD=center_of_hand-core_of_hand
    print(core_of_hand)
    print(elbow)
    print(SCD)
    print(FST)
    FstPoint, SndPoint, TrdPoint = elbow, core_of_hand , center_of_hand
    pr=0
    for i in range((FST.shape[0])):
        # print(np.linalg.norm(FST[i]))
        angles_l_hand.append(np.degrees(np.arcsin(-(FST[i] @ SCD[i])/(np.linalg.norm(FST[i])*np.linalg.norm(SCD[i])))))
        # if (abs(np.degrees(np.arcsin(-(FST[i] @ SCD[i])/(np.linalg.norm(FST[i])*np.linalg.norm(SCD[i]))))-pr))>10:
        #     angles_l_hand[i]=-angles_l_hand[i]
            
    
    # plot angles_r_hand
    plt.title('Отклонение кисти')
    plt.plot(angles_r_hand, label='Угол отклонения правой кисти (°)')
    angles_r_hand=np.array(angles_r_hand)
    plt.plot(angles_l_hand, label='Угол отклонения левой кисти (°)')
    plt.legend(loc='upper right')
    plt.xlim((0,1180))
    angles_l_hand=np.array(angles_l_hand)
    mr=[]
    ml=[]
    for start, end in ep:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        mx_r=max(abs(angles_r_hand[display_start:display_end]))
        mx_l=max(abs(angles_l_hand[display_start:display_end]))
        mr.append(mx_r)
        ml.append(mx_l)
    
        if display_start < display_end:
            plt.axvspan(display_start, display_end, color='gainsboro', alpha=0.5, label='_nolegend_')
    plt.ylim((-45,45))
    plt.show()
    print(np.mean(ml),np.mean(mr))


    angles_l_hand=[] 
    right_hand_idx = 1  
    elbow = keypoints_data[:, right_hand_idx, 4] 
    core_of_hand = keypoints_data[:, right_hand_idx, 2]
    center_of_hand = (keypoints_data[:, right_hand_idx, 5])
    center_of_hand_1 = (keypoints_data[:, right_hand_idx, 8])
    mxa=[]

    FstPoint, SndPoint, TrdPoint = elbow, core_of_hand , center_of_hand
    angles=calculate_angle_vectorized(FstPoint, SndPoint,center_of_hand_1, TrdPoint)
    for i in range(len(angles)):
        if angles[i]>85:
            angles[i]=angles[i-1]
    angles=np.array(angles)
    for start, end in ep:
        start_time, end_time = start , end 
        display_start = max(start_time, 0)
        display_end = min(end_time, 1200)
        # print(display_start ,display_end)
        mx=max(abs(angles[display_start:display_end]))
        mxa.append(mx)
        if display_start < display_end:
            plt.axvspan(display_start, display_end, color='gainsboro', alpha=0.5, label='_nolegend_')
    plt.title('Отведение большого пальца')
    plt.plot(angles, label='Угол отведения большого пальца (°)')
    plt.legend(loc='upper right')
    plt.xlim((0,1190))
    plt.ylim((0,90))
    plt.show()
    print(np.mean(mxa))