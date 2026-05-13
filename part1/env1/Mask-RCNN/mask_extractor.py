import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

class MaskExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load a pre-trained Mask R-CNN model (ResNet50 backbone)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO class IDs for dynamic objects (1: person, 2: bicycle, 3: car, etc.)
        self.dynamic_classes = [1, 2, 3, 4, 6, 8] 
        self.score_threshold = 0.5
        self.motion_threshold = 1.0 # Pixel displacement threshold for optical flow

    def get_masks(self, frame):
        """
        Extracts semantic masks for dynamic classes using Mask R-CNN.
        """
        # Convert BGR (OpenCV) to RGB, then to tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
            
        final_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for i in range(len(prediction['labels'])):
            label = prediction['labels'][i].item()
            score = prediction['scores'][i].item()
            
            if label in self.dynamic_classes and score > self.score_threshold:
                # Mask R-CNN outputs soft masks [0, 1], threshold at 0.5
                mask = prediction['masks'][i, 0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                final_mask = cv2.bitwise_or(final_mask, binary_mask)
                
        return final_mask

    def apply_optical_flow_filter(self, prev_frame, curr_frame, curr_mask):
        """
        Filters out static objects using Lucas-Kanade Sparse Optical Flow.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Find good features to track within the detected mask
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=curr_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        if p0 is None:
            return np.zeros_like(curr_mask) # No features, treat as static
            
        # Calculate optical flow using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(15, 15), maxLevel=2)
        
        if p1 is None:
            return np.zeros_like(curr_mask)
            
        # Select good points and calculate movement magnitude
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) == 0:
            return np.zeros_like(curr_mask)
            
        distances = np.linalg.norm(good_new - good_old, axis=1)
        mean_movement = np.mean(distances)
        
        # If movement is significant, keep the mask; otherwise, it's static
        if mean_movement > self.motion_threshold:
            # Advanced Idea: Apply Dilation to cover motion blur
            kernel = np.ones((15, 15), np.uint8)
            dilated_mask = cv2.dilate(curr_mask, kernel, iterations=1)
            return dilated_mask
        else:
            return np.zeros_like(curr_mask)