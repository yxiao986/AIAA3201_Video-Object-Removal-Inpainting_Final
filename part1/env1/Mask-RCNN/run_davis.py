import os
import sys
import glob
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import evaluate_mask_quality
from mask_extractor import MaskExtractor
from inpainter import Inpainter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Part 1 Baseline on the full DAVIS dataset")
    parser.add_argument("--davis_root", type=str, default="../data/DAVIS", 
                        help="Path to the DAVIS root directory")
    parser.add_argument("--output_dir", type=str, default="../results/part1_davis_eval", 
                        help="Directory to save metrics and masks")
    parser.add_argument("--run_inpainting", action="store_true", 
                        help="Flag to also run the inpainting process (Warning: Very time-consuming for the whole dataset)")
    return parser.parse_args()

def load_frames(folder_path, is_mask=False):
    """Loads images from a directory, sorted alphabetically."""
    if not folder_path or not os.path.exists(folder_path):
        return [], []
    
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + 
                        glob.glob(os.path.join(folder_path, '*.png')))
    
    frames = []
    for p in file_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        frames.append(img)
    return frames, file_paths

def main():
    args = parse_args()
    
    # Check your specific directory structure shown in the screenshot
    jpeg_dir = os.path.join(args.davis_root, "JPEGImages", "480p")
    anno_dir = os.path.join(args.davis_root, "Annotations", "480p")
    
    if not os.path.exists(jpeg_dir):
        print(f"[Error] DAVIS JPEGImages directory not found at {jpeg_dir}")
        return
        
    # Get all sequences dynamically
    sequences = sorted([d for d in os.listdir(jpeg_dir) if os.path.isdir(os.path.join(jpeg_dir, d))])
    print(f"Found {len(sequences)} video sequences in the DAVIS dataset.")
    
    extractor = MaskExtractor()
    inpainter = Inpainter(temporal_window=15) if args.run_inpainting else None
    
    all_metrics = {}
    global_jm = []
    global_jr = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for seq in sequences:
        print(f"\n{'-'*40}\nProcessing Sequence: {seq}\n{'-'*40}")
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_anno = os.path.join(anno_dir, seq)
        
        seq_out_dir = os.path.join(args.output_dir, seq)
        mask_out_dir = os.path.join(seq_out_dir, "masks")
        os.makedirs(mask_out_dir, exist_ok=True)
        
        frames, frame_paths = load_frames(seq_jpeg)
        gt_masks, _ = load_frames(seq_anno, is_mask=True)
        
        if not frames:
            continue
            
        pred_masks = []
        for i, frame in enumerate(tqdm(frames, desc=f"Extracting {seq}")):
            raw_mask = extractor.get_masks(frame)
            if i == 0:
                final_mask = raw_mask
            else:
                final_mask = extractor.apply_optical_flow_filter(frames[i-1], frame, raw_mask)
            pred_masks.append(final_mask)
            
            out_name = os.path.basename(frame_paths[i]).replace('.jpg', '.png')
            cv2.imwrite(os.path.join(mask_out_dir, out_name), final_mask)
            
        # Evaluation against Annotations
        if len(gt_masks) == len(pred_masks):
            metrics = evaluate_mask_quality(pred_masks, gt_masks)
            all_metrics[seq] = metrics
            global_jm.append(metrics['J_M'])
            global_jr.append(metrics['J_R'])
            print(f"[{seq}] Result -> J_M: {metrics['J_M']:.4f}, J_R: {metrics['J_R']:.4f}")
        else:
            print(f"[Warning] Ground Truth mismatch for {seq}. Skipping metrics.")
            
        # Optional full inpainting
        if args.run_inpainting:
            inpaint_out_dir = os.path.join(seq_out_dir, "inpainted")
            os.makedirs(inpaint_out_dir, exist_ok=True)
            inpainted_frames = inpainter.inpaint(frames, pred_masks)
            for i, res_frame in enumerate(tqdm(inpainted_frames, desc=f"Inpainting {seq}")):
                out_name = os.path.basename(frame_paths[i])
                cv2.imwrite(os.path.join(inpaint_out_dir, out_name), res_frame)
                
    # Aggregate Global Summary
    if global_jm:
        avg_jm = np.mean(global_jm)
        avg_jr = np.mean(global_jr)
        all_metrics["AVERAGE_DATASET_SCORE"] = {"J_M": avg_jm, "J_R": avg_jr}
        
        print(f"\n{'='*50}")
        print(f" 🏆 DAVIS FULL DATASET EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f" Total Sequences Processed: {len(global_jm)}")
        print(f" Global Average J_M (Mean IoU): {avg_jm:.4f}")
        print(f" Global Average J_R (Recall):   {avg_jr:.4f}")
        print(f"{'='*50}")
        
        # Save overarching metrics to the root output folder
        with open(os.path.join(args.output_dir, "davis_global_metrics.json"), 'w') as f:
            json.dump(all_metrics, f, indent=4)

if __name__ == '__main__':
    main()