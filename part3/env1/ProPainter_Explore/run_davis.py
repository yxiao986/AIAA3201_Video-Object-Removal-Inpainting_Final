import os
import sys
import glob
import cv2
import numpy as np
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root_dir)

from part3.env1.ProPainter_Explore.main import run_propainter, run_diffueraser_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: Upper-Bound Generative Inpainting on DAVIS (GT Injection)")
    parser.add_argument("--davis_root", type=str, default=os.path.join(project_root_dir, "data", "DAVIS"), 
                        help="Path to the DAVIS root directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root_dir, "results", "part3_davis_eval"), 
                        help="Directory to save comparison results")
    parser.add_argument("--method", type=str, choices=['baseline', 'diffueraser', 'sd2d'], required=True,
                        help="Which method to run (baseline or diffueraser)")
    parser.add_argument("--target_seqs", type=str, nargs="+", default=[],
                        help="Specific sequences to evaluate. If empty, runs ALL available sequences.")
    return parser.parse_args()

def main():
    args = parse_args()
    jpeg_dir = os.path.join(args.davis_root, "JPEGImages", "480p")
    anno_dir = os.path.join(args.davis_root, "Annotations", "480p")
    
    if not os.path.exists(jpeg_dir):
        print(f"[Error] DAVIS directory not found at {jpeg_dir}")
        return

    sequences = args.target_seqs if args.target_seqs else sorted([d for d in os.listdir(jpeg_dir) if os.path.isdir(os.path.join(jpeg_dir, d))])
        
    print(f"\n{'='*70}")
    print(f" Part 3: Upper-Bound Evaluation (GT Injection)")
    print(f" Executing Method: {args.method.upper()}")
    print(f" Sequences to process: {len(sequences)}")
    print(f"{'='*70}\n")
    
    for seq in sequences:
        print(f"\n>> Processing Sequence: [{seq}]")
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_anno = os.path.join(anno_dir, seq)
        
        seq_out_dir = os.path.join(args.output_dir, seq)
        dilated_mask_dir = os.path.join(seq_out_dir, "gt_masks_dilated")
        method_out_dir = os.path.join(seq_out_dir, args.method)
        
        os.makedirs(dilated_mask_dir, exist_ok=True)
        os.makedirs(method_out_dir, exist_ok=True)
        

        if not glob.glob(os.path.join(dilated_mask_dir, "*.png")):
            print("   -> Injecting and dilating Ground Truth masks...")
            gt_mask_files = sorted(glob.glob(os.path.join(seq_anno, "*.png")))
            if not gt_mask_files:
                print(f"   [Error] No GT masks found for {seq}. Skipping.")
                continue
            kernel = np.ones((5, 5), np.uint8)
            for f in gt_mask_files:
                mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(dilated_mask_dir, os.path.basename(f)), cv2.dilate(mask, kernel, iterations=1))

        print(f"   -> Executing {args.method.upper()}...")
        if args.method == 'baseline':
            run_propainter(seq_jpeg, dilated_mask_dir, method_out_dir)
        elif args.method == 'diffueraser':
            run_diffueraser_inference(seq_jpeg, dilated_mask_dir, method_out_dir)
        elif args.method == 'sd2d':
            print("   [Warning] SD2D batch processing skipped in this script. Use main.py.")

        print(f"   [Success] Sequence '{seq}' completed.")

if __name__ == '__main__':
    main()