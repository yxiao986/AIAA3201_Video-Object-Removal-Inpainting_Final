import os
import sys
import glob
import cv2
import json
import argparse
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import evaluate_mask_quality
from mask_extractor import MaskExtractor
from inpainter import Inpainter

def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 1 Baseline Pipeline")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the dataset (e.g., tennis, bmx-trees, wild_video)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the input video frames")
    parser.add_argument("--gt_mask_dir", type=str, default=None, 
                        help="Path to the ground truth masks (optional, for evaluation)")
    parser.add_argument("--output_base_dir", type=str, default="../results/part1_baseline", 
                        help="Base directory to save results")
    return parser.parse_args()

def load_frames(folder_path, is_mask=False):
    """Loads images from a directory, sorted alphabetically."""
    if not folder_path or not os.path.exists(folder_path):
        return []
    
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + 
                        glob.glob(os.path.join(folder_path, '*.png')))
    
    frames = []
    for p in file_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        frames.append(img)
    return frames, file_paths

def main():
    args = parse_args()
    
    # Setup output directories
    dataset_out_dir = os.path.join(args.output_base_dir, args.dataset_name)
    mask_out_dir = os.path.join(dataset_out_dir, "masks")
    inpaint_out_dir = os.path.join(dataset_out_dir, "inpainted")
    
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(inpaint_out_dir, exist_ok=True)
    
    print(f"[{args.dataset_name}] Loading frames from {args.data_dir}...")
    frames, frame_paths = load_frames(args.data_dir)
    if not frames:
        print("No frames found! Exiting.")
        return

    # --- Step 1: Mask Extraction ---
    extractor = MaskExtractor()
    pred_masks = []
    
    print(f"[{args.dataset_name}] Extracting masks...")
    for i, frame in enumerate(tqdm(frames)):
        raw_mask = extractor.get_masks(frame)
        
        if i == 0:
            final_mask = raw_mask
        else:
            final_mask = extractor.apply_optical_flow_filter(frames[i-1], frame, raw_mask)
            
        pred_masks.append(final_mask)
        
        # Save generated mask
        out_name = os.path.basename(frame_paths[i]).replace('.jpg', '.png')
        cv2.imwrite(os.path.join(mask_out_dir, out_name), final_mask)

    # --- Step 2: Evaluation (If GT masks exist) ---
    if args.gt_mask_dir:
        print(f"[{args.dataset_name}] Loading GT masks for evaluation...")
        gt_masks, _ = load_frames(args.gt_mask_dir, is_mask=True)
        
        if len(gt_masks) == len(pred_masks):
            metrics = evaluate_mask_quality(pred_masks, gt_masks)
            print(f"--- Evaluation Results ---")
            print(f"J_M (Mean IoU): {metrics['J_M']:.4f}")
            print(f"J_R (IoU Recall): {metrics['J_R']:.4f}")
            
            # Save metrics to JSON
            metrics_path = os.path.join(dataset_out_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        else:
            print("Warning: Number of GT masks does not match frame count. Skipping evaluation.")

    # --- Step 3: Inpainting ---
    print(f"[{args.dataset_name}] Starting Temporal + Spatial Inpainting...")
    inpainter = Inpainter(temporal_window=15)
    inpainted_frames = inpainter.inpaint(frames, pred_masks)
    
    print(f"[{args.dataset_name}] Saving inpainted results...")
    for i, res_frame in enumerate(tqdm(inpainted_frames)):
        out_name = os.path.basename(frame_paths[i])
        cv2.imwrite(os.path.join(inpaint_out_dir, out_name), res_frame)
        
    print(f"[{args.dataset_name}] Pipeline completed successfully!\n")

if __name__ == '__main__':
    main()