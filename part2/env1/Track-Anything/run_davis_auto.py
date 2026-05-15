import os
import sys
import glob
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, "..", "..", ".."))

track_anything_dir = os.path.join(project_root_dir, "third_party", "Track-Anything")
tracker_dir = os.path.join(track_anything_dir, "tracker")

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
if track_anything_dir not in sys.path:
    sys.path.insert(0, track_anything_dir)  
if tracker_dir not in sys.path:
    sys.path.insert(0, tracker_dir)         

from tracker.base_tracker import BaseTracker
from utils.metrics import evaluate_mask_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Fully Automated Part 2 Evaluation on DAVIS (First-Frame GT Injection)")
    parser.add_argument("--davis_root", type=str, default=os.path.join(project_root_dir, "data", "DAVIS"), 
                        help="Path to the DAVIS root directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root_dir, "results", "part2_davis_auto_eval"), 
                        help="Directory to save automated metrics and masks")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--xmem_checkpoint", type=str,
                        default=os.path.join(track_anything_dir, "checkpoints", "XMem-s012.pth"),
                        help="Path to the XMem checkpoint used by Track-Anything.")
    return parser.parse_args()

def load_frames(folder_path, is_mask=False):
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
    
    jpeg_dir = os.path.join(args.davis_root, "JPEGImages", "480p")
    anno_dir = os.path.join(args.davis_root, "Annotations", "480p")
    
    if not os.path.exists(jpeg_dir):
        print(f"[Error] DAVIS directory not found at {jpeg_dir}")
        return

    sequences = sorted([d for d in os.listdir(jpeg_dir) if os.path.isdir(os.path.join(jpeg_dir, d))])
        
    print(f"\n{'='*70}")
    print(f" Part 2 Automated Evaluation (First-Frame GT Injection)")
    print(f" Sequences to process: {len(sequences)}")
    print(f"{'='*70}\n")
    
    xmem_checkpoint = os.path.abspath(args.xmem_checkpoint)
    if not os.path.exists(xmem_checkpoint):
        print(f"[Error] XMem checkpoint not found at {xmem_checkpoint}")
        return
        
    original_cwd = os.getcwd()  
    try:
        os.chdir(track_anything_dir) 
        tracker = BaseTracker(xmem_checkpoint, device=args.device)
        
    finally:
        os.chdir(original_cwd)
    
    all_metrics = {}
    global_jm, global_jr = [], []
    os.makedirs(args.output_dir, exist_ok=True)
    
    for seq in sequences:
        print(f"\n>> Auto-Tracking Sequence: [{seq}]")
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_anno = os.path.join(anno_dir, seq)
        
        seq_out_dir = os.path.join(args.output_dir, seq)
        mask_out_dir = os.path.join(seq_out_dir, "masks")
        os.makedirs(mask_out_dir, exist_ok=True)
        
        frames, frame_paths = load_frames(seq_jpeg)
        gt_masks, _ = load_frames(seq_anno, is_mask=True)
        
        if not frames or len(frames) != len(gt_masks):
            print(f"   [Warning] Data mismatch for {seq}. Skipping.")
            continue
            
        pred_masks = []
        
        # 1. Clear any existing memory in the tracker to ensure a fresh start for each sequence
        tracker.clear_memory()
        
        # 2. Inject the first frame and its GT mask into the tracker as the initial template
        first_frame = frames[0]
        first_mask = gt_masks[0]

        xmem_prompt_mask = (first_mask > 127).astype(np.uint8)
        
        # Use the first GT mask as the initial reference for tracking
        tracker.track(first_frame, xmem_prompt_mask)
        pred_masks.append(first_mask) 
        cv2.imwrite(os.path.join(mask_out_dir, os.path.basename(frame_paths[0]).replace('.jpg', '.png')), first_mask)
        
        # 3. Iteratively track the object in subsequent frames using the tracker's internal mechanism (without any further GT injection)
        for i in tqdm(range(1, len(frames)), desc=f"Propagating {seq}"):
            track_result = tracker.track(frames[i])
            pred_mask = track_result[0] if isinstance(track_result, tuple) else track_result
            pred_mask_255 = (pred_mask * 255).astype(np.uint8) if pred_mask.max() <= 1 else pred_mask.astype(np.uint8)
            pred_masks.append(pred_mask_255)
            
            out_name = os.path.basename(frame_paths[i]).replace('.jpg', '.png')
            cv2.imwrite(os.path.join(mask_out_dir, out_name), pred_mask)
            
        # 4. Evaluate the predicted masks against the GT masks using the provided metrics
        metrics = evaluate_mask_quality(pred_masks, gt_masks)
        all_metrics[seq] = metrics
        global_jm.append(metrics['J_M'])
        global_jr.append(metrics['J_R'])
        print(f"   [Result] J_M: {metrics['J_M']:.4f}, J_R: {metrics['J_R']:.4f}")

    # ==========================================
    # 5. Global Summary and User Prompt for Manual Mask Generation (if needed)
    # ==========================================
    if global_jm:
        avg_jm, avg_jr = np.mean(global_jm), np.mean(global_jr)
        all_metrics["AVERAGE_DATASET_SCORE"] = {"J_M": avg_jm, "J_R": avg_jr}
        
        print(f"\n{'='*70}")
        print(f" TRACK-ANYTHING FULL DAVIS EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f" Sequences Processed: {len(global_jm)}")
        print(f" Global Average J_M (Mean IoU): {avg_jm:.4f}")
        print(f" Global Average J_R (Recall):   {avg_jr:.4f}")
        print(f"{'='*70}")
        
        with open(os.path.join(args.output_dir, "track_anything_davis_global.json"), 'w') as f:
            json.dump(all_metrics, f, indent=4)

if __name__ == '__main__':
    main()
