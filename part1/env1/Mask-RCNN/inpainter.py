import cv2
import numpy as np

class Inpainter:
    def __init__(self, temporal_window=10):
        # How many frames forward/backward to look for clean pixels
        self.temporal_window = temporal_window

    def temporal_background_propagation(self, frames, masks, target_idx):
        """
        Advanced Idea: Borrows clean pixels from adjacent frames.
        """
        target_frame = frames[target_idx].copy()
        target_mask = masks[target_idx].copy()
        
        h, w = target_mask.shape
        # Create a boolean mask where True means the pixel needs to be fixed
        pixels_to_fix = target_mask > 127
        
        # Search window
        start_idx = max(0, target_idx - self.temporal_window)
        end_idx = min(len(frames), target_idx + self.temporal_window + 1)
        
        for idx in range(start_idx, end_idx):
            if idx == target_idx:
                continue
                
            search_mask = masks[idx]
            search_frame = frames[idx]
            
            # Find pixels that need fixing in target AND are clean in the search frame
            clean_in_search = search_mask < 127
            can_borrow = pixels_to_fix & clean_in_search
            
            # Copy clean pixels over
            target_frame[can_borrow] = search_frame[can_borrow]
            
            # Update the pixels_to_fix status (they are now fixed)
            pixels_to_fix[can_borrow] = False
            
            if not np.any(pixels_to_fix):
                break # All pixels are fixed, exit early
                
        # Update the mask to only represent pixels that still need spatial inpainting
        remaining_mask = np.zeros_like(target_mask)
        remaining_mask[pixels_to_fix] = 255
        
        return target_frame, remaining_mask

    def inpaint(self, frames, masks):
        """
        Combines Temporal Propagation with Spatial Inpainting fallback.
        """
        results = []
        for i in range(len(frames)):
            # Step 1: Temporal Propagation
            temp_inpainted, remaining_mask = self.temporal_background_propagation(frames, masks, i)
            
            # Step 2: Fallback to spatial cv2.inpaint for any remaining holes
            if np.any(remaining_mask > 0):
                # Using Navier-Stokes algorithm (cv2.INPAINT_NS)
                final_inpainted = cv2.inpaint(temp_inpainted, remaining_mask, 3, cv2.INPAINT_NS)
            else:
                final_inpainted = temp_inpainted
                
            results.append(final_inpainted)
            
        return results