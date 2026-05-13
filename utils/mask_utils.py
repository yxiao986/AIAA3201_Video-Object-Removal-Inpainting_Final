import cv2
import numpy as np
import random
import os

def generate_random_stationary_mask(height, width, num_strokes=6, base_thickness=50):
    """
    Generates a stationary mask with random strokes and circles.
    1 represents masked area (255), 0 represents background.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for _ in range(num_strokes):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        thickness = random.randint(base_thickness // 2, int(base_thickness * 1.5))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
    for _ in range(num_strokes // 2):
        cx, cy = random.randint(0, width), random.randint(0, height)
        radius = random.randint(base_thickness // 2, base_thickness)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
    return mask

def save_stationary_mask_sequence(mask, output_dir, num_frames, frame_names):
    """
    Saves the identical mask for all frames to simulate a stationary occlusion.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_paths = []
    for name in frame_names:
        out_name = os.path.splitext(name)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, mask)
        mask_paths.append(out_path)
    return mask_paths