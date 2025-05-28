import cv2
import mediapipe as mp
import numpy as np
import config 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeypointDetector:
    """
    Detects 2D hand landmarks using MediaPipe and estimates 3D coordinates
    using a depth map.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.MP_STATIC_IMAGE_MODE,
            max_num_hands=config.MP_MAX_NUM_HANDS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE
        )
        self.connections = self.mp_hands.HAND_CONNECTIONS
        self.fx = config.FX
        self.fy = config.FY
        self.cx = config.CX
        self.cy = config.CY
        self.depth_scale = config.DEPTH_SCALE
        self.depth_radius = config.DEPTH_SEARCH_RADIUS
        logging.info("KeypointDetector initialized.")

    def _detect_hands_2d(self, rgb_image):
        """Detects 2D landmarks in a single RGB image."""
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        return results, rgb_image.shape 

    def _get_3d_coords_from_depth(self, landmarks_2d, depth_image, img_height, img_width):
        """
        Estimates 3D coordinates for landmarks using depth neighborhood.
        Returns coordinates in meters.
        """
        landmarks_3d = np.full((config.NUM_LANDMARKS_MEDIAPIPE, 3), np.nan)

        if depth_image is None:
            logging.error("Depth image is None, cannot calculate 3D coordinates.")
            return landmarks_3d

        depth_height, depth_width = depth_image.shape

        for i, landmark in enumerate(landmarks_2d.landmark):
            
            u = int(landmark.x * img_width)
            v = int(landmark.y * img_height)

            depth_values = []
            coords_3d_list = []


            for dv in range(-self.depth_radius, self.depth_radius + 1):
                for du in range(-self.depth_radius, self.depth_radius + 1):
                    current_v, current_u = v + dv, u + du


                    if 0 <= current_v < depth_height and 0 <= current_u < depth_width:
                        depth = depth_image[current_v, current_u]
                        if depth > 0: # Ensure depth is valid
                            depth_m = depth / self.depth_scale 
                            x_3d = (current_u - self.cx) * depth_m / self.fx
                            y_3d = (current_v - self.cy) * depth_m / self.fy
                            z_3d = depth_m
                            depth_values.append(depth_m)
                            coords_3d_list.append((x_3d, y_3d, z_3d))

            if coords_3d_list:

                all_x = [c[0] for c in coords_3d_list]
                all_y = [c[1] for c in coords_3d_list]
                all_z = [c[2] for c in coords_3d_list]

                median_x = np.median(all_x)
                median_y = np.median(all_y)
                median_z = np.min(all_z) # Using median for Z


                landmarks_3d[i, 0] = median_x * 1000 
                landmarks_3d[i, 1] = median_y * 1000 
                landmarks_3d[i, 2] = -median_z * 1000 

        return landmarks_3d


    def process_frame(self, color_image_path, depth_image_path):
        """
        Processes a single pair of color and depth images.

        Returns:
            list[np.ndarray]: A list containing 3D keypoints (shape [21, 3])
                              for each detected hand (in mm, Z flipped).
                              Returns an empty list if no hands detected or error.
        """
        rgb_image = cv2.imread(color_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

        if rgb_image is None:
            logging.error(f"Failed to load color image: {color_image_path}")
            return []
        if depth_image is None:
            logging.warning(f"Failed to load depth image: {depth_image_path}. Proceeding without 3D.")

        results_2d, img_shape = self._detect_hands_2d(rgb_image)
        img_height, img_width, _ = img_shape

        all_hands_3d = []
        all_hands_2d = []
        if results_2d.multi_hand_landmarks:
            num_detected = len(results_2d.multi_hand_landmarks)
            for hand_landmarks_2d in results_2d.multi_hand_landmarks:
                landmarks_3d = self._get_3d_coords_from_depth(
                    hand_landmarks_2d, depth_image, img_height, img_width
                )
                all_hands_3d.append(landmarks_3d)
                all_hands_2d.append(hand_landmarks_2d)

        output_hands = []
        output_hands_2d = []
        for i in range(config.NUM_HANDS):
            if i < len(all_hands_3d):
                output_hands.append(all_hands_3d[i])
                output_hands_2d.append(all_hands_2d[i])
            else:
                output_hands.append(np.full((config.NUM_LANDMARKS_MEDIAPIPE, 3), np.nan))
                output_hands_2d.append(np.full((config.NUM_LANDMARKS_MEDIAPIPE, 3), np.nan))

        return output_hands, output_hands_2d 
    def close(self):
        self.hands.close()
        logging.info("KeypointDetector closed.")