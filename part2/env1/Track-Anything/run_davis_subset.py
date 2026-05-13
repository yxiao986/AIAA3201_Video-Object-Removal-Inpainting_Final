import os
import sys
import glob
import cv2
import json
import argparse
import subprocess
import numpy as np

# Dynamically get absolute paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root_dir)

from utils.metrics import evaluate_mask_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Part 2 SOTA Pipeline on specific DAVIS sequences")
    parser.add_argument("--davis_root", type=str, default=os.path.join(project_root_dir, "data", "DAVIS"), 
                        help="Path to the DAVIS root directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root_dir, "results", "part2_davis_eval"), 
                        help="Directory to save results")
    parser.add_argument("--target_seqs", type=str, nargs="+", required=True,
                        help="Specific sequences to evaluate (e.g., skate-jump camel). Must be generated via UI first.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    jpeg_dir = os.path.join(args.davis_root, "JPEGImages", "480p")
    anno_dir = os.path.join(args.davis_root, "Annotations", "480p")
    propainter_dir = os.path.join(project_root_dir, "third_party", "ProPainter")
    
    if not os.path.exists(jpeg_dir):
        print(f"[Error] DAVIS directory not found at {jpeg_dir}")
        return

    print(f"Preparing to evaluate interactive SOTA pipeline on sequences: {args.target_seqs}")
    
    all_metrics = {}
    global_jm, global_jr = [], []
    
    # Custom environment for ProPainter
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = project_root_dir + os.pathsep + custom_env.get("PYTHONPATH", "")

    for seq in args.target_seqs:
        print(f"\n{'-'*50}\nProcessing SOTA Pipeline for Sequence: [{seq}]\n{'-'*50}")
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_anno = os.path.join(anno_dir, seq)
        
        seq_out_dir = os.path.join(args.output_dir, seq)
        mask_out_dir = os.path.join(seq_out_dir, "masks")
        inpaint_out_dir = os.path.join(seq_out_dir, "inpainted")
        
        os.makedirs(mask_out_dir, exist_ok=True)
        os.makedirs(inpaint_out_dir, exist_ok=True)
        
        # --- Step 1: Interactive Mask Evaluation ---
        existing_masks = glob.glob(os.path.join(mask_out_dir, "*.png"))
        if not existing_masks:
            print(f"-> [Error] No Track-Anything masks found in {mask_out_dir}.")
            print(f"   Please use Track-Anything UI (app.py) to generate and save them here first.")
            continue
            
        print(f"-> Found {len(existing_masks)} interactively generated masks.")
        
        pred_paths = sorted(existing_masks)
        gt_paths = sorted(glob.glob(os.path.join(seq_anno, "*.png")))
        
        if len(pred_paths) == len(gt_paths):
            pred_masks = [cv2.imread(p, 0) for p in pred_paths]
            gt_masks = [cv2.imread(p, 0) for p in gt_paths]
            metrics = evaluate_mask_quality(pred_masks, gt_masks)
            all_metrics[seq] = metrics
            global_jm.append(metrics['J_M'])
            global_jr.append(metrics['J_R'])
            print(f"   Interactive Mask Quality -> J_M: {metrics['J_M']:.4f}, J_R: {metrics['J_R']:.4f}")
        else:
            print(f"-> [Warning] Number of generated masks ({len(pred_paths)}) does not match GT ({len(gt_paths)}). Skipping metrics.")

        # --- Step 2: Mask Refinement (Dilation) ---
        print(f"-> Applying morphological dilation to mask boundaries...")
        kernel = np.ones((5, 5), np.uint8) 
        for mask_path in pred_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            cv2.imwrite(mask_path, dilated_mask)

        # --- Step 3: ProPainter Inpainting ---
        print(f"-> Running ProPainter for {seq}...")
        try:
            subprocess.run([
                sys.executable, "inference_propainter.py",
                "--video", os.path.abspath(seq_jpeg),
                "--mask", os.path.abspath(mask_out_dir),
                "--output", os.path.abspath(inpaint_out_dir)
            ], check=True, cwd=propainter_dir, env=custom_env, stdout=subprocess.DEVNULL)
            print(f"-> Successfully inpainted {seq}.")
        except subprocess.CalledProcessError as e:
            print(f"[Error] ProPainter failed for {seq}: {e}")

    # Aggregate global metrics
    if global_jm:
        avg_jm, avg_jr = np.mean(global_jm), np.mean(global_jr)
        all_metrics["AVERAGE_SUBSET_SCORE"] = {"J_M": avg_jm, "J_R": avg_jr}
        
        print(f"\n{'='*50}")
        print(f"INTERACTIVE TRACKING SUMMARY (Subset)")
        print(f"{'='*50}")
        print(f" Sequences Processed: {len(global_jm)}")
        print(f" Average J_M (Mean IoU): {avg_jm:.4f}")
        print(f" Average J_R (Recall):   {avg_jr:.4f}")
        print(f"{'='*50}")
        
        with open(os.path.join(args.output_dir, "track_anything_subset_metrics.json"), 'w') as f:
            json.dump(all_metrics, f, indent=4)

if __name__ == '__main__':
    main()