import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import config 
import logging


connections_extended = config.SKELETON_CONNECTIONS + [(0, config.NUM_LANDMARKS_MEDIAPIPE)] 

def setup_skeleton_plot(ax):
    """Sets up the common properties for the 3D skeleton plot."""
    if config.VISUALIZATION_XLIM: ax.set_xlim(config.VISUALIZATION_XLIM)
    if config.VISUALIZATION_YLIM: ax.set_ylim(config.VISUALIZATION_YLIM)
    if config.VISUALIZATION_ZLIM: ax.set_zlim(config.VISUALIZATION_ZLIM)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True 
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.6)) 

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    ax.view_init(elev=config.VISUALIZATION_ELEV, azim=config.VISUALIZATION_AZIM)

def plot_hand_skeleton(ax, hand_keypoints, color='r', linewidth=1.5, marker='o', markersize=3):
    """Plots a single hand skeleton on the given Axes3D."""
    num_landmarks_in_data = hand_keypoints.shape[0]

    valid_mask = ~np.isnan(hand_keypoints).any(axis=1)
    valid_points = hand_keypoints[valid_mask]

    if valid_points.shape[0] > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                   marker=marker, s=markersize**2, c=color, depthshade=True) 


    for start_idx, end_idx in connections_extended:

        if 0 <= start_idx < num_landmarks_in_data and 0 <= end_idx < num_landmarks_in_data:
            p_start = hand_keypoints[start_idx]
            p_end = hand_keypoints[end_idx]

            if not np.isnan(p_start).any() and not np.isnan(p_end).any():
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
                        color + '-', linewidth=linewidth) 

def visualize_frame(keypoints_frame_data, frame_index, title_suffix=""):
    """
    Visualizes the hand skeletons for a single frame.

    Args:
        keypoints_frame_data (np.ndarray): Keypoints for the frame,
                                           shape (num_hands, num_landmarks, 3).
        frame_index (int): Index of the frame for title and saving.
        title_suffix (str): Text to append to the plot title.
    """
    fig = plt.figure(figsize=(10, 7)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.clear() 
    setup_skeleton_plot(ax)

    num_hands = keypoints_frame_data.shape[0]
    colors = ['b', 'r'] 

    for i in range(num_hands):
        hand_data = keypoints_frame_data[i]
        plot_hand_skeleton(ax, hand_data, color=colors[i % len(colors)])

    ax.set_title(f'Frame {frame_index} {title_suffix}')

    if config.VISUALIZATION_SAVE_IMAGES:
        filename = f'frame_{frame_index:05d}{title_suffix.lower().replace(" ", "_")}.png'
        save_path = os.path.join(config.VISUALIZATION_DIR, filename)
        try:
            plt.savefig(save_path, dpi=config.VISUALIZATION_DPI, bbox_inches='tight')
            logging.debug(f"Saved visualization: {save_path}")
        except Exception as e:
            logging.error(f"Error saving visualization {save_path}: {e}")

    if config.VISUALIZATION_PAUSE_DURATION > 0:
        plt.pause(config.VISUALIZATION_PAUSE_DURATION)

    plt.close(fig) 