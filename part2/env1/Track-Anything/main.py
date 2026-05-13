import os
import sys
import glob
import cv2
import json
import argparse
import subprocess

# Dynamically get absolute paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

# Add project root to sys.path to import utils
sys.path.append(project_root_dir)
from utils.metrics import evaluate_mask_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 2 SOTA Pipeline (Interactive Mode)")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, default=None)
    parser.add_argument("--output_base_dir", type=str, default=os.path.join(project_root_dir, "results", "part2_sota"))
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Absolute paths for third-party tools
    track_anything_dir = os.path.join(project_root_dir, "third_party", "Track-Anything")
    propainter_dir = os.path.join(project_root_dir, "third_party", "ProPainter")
    
    # Check dependencies
    if not os.path.isdir(propainter_dir) or not os.path.isdir(track_anything_dir):
        print("[Error] third_party directories not found. Please ensure they are placed in the project root.")
        return

    data_dir_abs = os.path.abspath(args.data_dir)
    gt_mask_dir_abs = os.path.abspath(args.gt_mask_dir) if args.gt_mask_dir else None
    
    # Output directories
    dataset_out_dir = os.path.abspath(os.path.join(args.output_base_dir, args.dataset_name))
    mask_out_dir = os.path.join(dataset_out_dir, "masks")
    inpaint_out_dir = os.path.join(dataset_out_dir, "inpainted")
    
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(inpaint_out_dir, exist_ok=True)

    # --- Step 1: Interactive Mask Generation ---
    print(f"\n[{args.dataset_name}] Step 1: Checking for Track-Anything Masks...")
    existing_masks = glob.glob(os.path.join(mask_out_dir, "*.png"))
    
    if not existing_masks:
        print("\n" + "="*60)
        print(" ACTION REQUIRED: HIGH-QUALITY MASK GENERATION")
        print("="*60)
        print("To achieve SOTA results, please generate masks interactively:")
        print("1. Open a NEW terminal.")
        print(f"2. Navigate to: {track_anything_dir}")
        print("3. Run: python app.py")
        print("4. Open the Web UI, upload the first frame, and click the target object.")
        print("5. Click 'Tracking' and save the generated masks to:")
        print(f"   --> {mask_out_dir}")
        print("="*60)
        input(">> Press [ENTER] only after you have saved all masks to the folder above...")
        
        # Re-check masks after user presses Enter
        existing_masks = glob.glob(os.path.join(mask_out_dir, "*.png"))
        if not existing_masks:
            print("[Error] No masks found. Exiting pipeline.")
            return
    
    print(f"-> Found {len(existing_masks)} masks. Proceeding to evaluation.")

    # --- Step 1.5: Mask Refinement (Dilation) ---
    print(f"\n[{args.dataset_name}] Step 1.5: Applying morphological dilation to masks...")
    import numpy as np
    
    # Use a 5x5 kernel. You can increase to (7, 7) if the ghosting is severe.
    kernel = np.ones((5, 5), np.uint8) 
    refined_count = 0
    
    for mask_path in existing_masks:
        # Read the mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            # Apply dilation to expand the mask boundaries
            # This covers motion blur and prevents artifacts like the "ghost racket"
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Overwrite the original mask with the dilated version
            cv2.imwrite(mask_path, dilated_mask)
            refined_count += 1
            
    print(f"-> Successfully dilated {refined_count} masks.")

    # --- Step 2: Mask Evaluation ---
    if gt_mask_dir_abs:
        print(f"\n[{args.dataset_name}] Step 2: Evaluating SOTA mask quality...")
        pred_masks_paths = sorted(glob.glob(os.path.join(mask_out_dir, "*.png")))
        gt_masks_paths = sorted(glob.glob(os.path.join(gt_mask_dir_abs, "*.png")))
        
        if len(pred_masks_paths) == len(gt_masks_paths) and len(gt_masks_paths) > 0:
            pred_masks = [cv2.imread(p, 0) for p in pred_masks_paths]
            gt_masks = [cv2.imread(p, 0) for p in gt_masks_paths]
            metrics = evaluate_mask_quality(pred_masks, gt_masks)
            
            with open(os.path.join(dataset_out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"-> Evaluation Done: J_M={metrics['J_M']:.4f}, J_R={metrics['J_R']:.4f}")
        else:
            print("-> [Warning] Number of generated masks does not match GT masks. Skipping evaluation.")

    # --- Step 3: Video Inpainting via ProPainter ---
    print(f"\n[{args.dataset_name}] Step 3: Inpainting using ProPainter...")
    propainter_script = os.path.join(propainter_dir, "inference_propainter.py")
    
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = project_root_dir + os.pathsep + custom_env.get("PYTHONPATH", "")
    
    try:
        subprocess.run([
            sys.executable, propainter_script,
            "--video", data_dir_abs,
            "--mask", mask_out_dir,
            "--output", inpaint_out_dir
        ], check=True, cwd=propainter_dir, env=custom_env) 
        print(f"\n[{args.dataset_name}] Pipeline completed successfully! Results saved to {dataset_out_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] ProPainter failed during execution: {e}")

if __name__ == '__main__':
    main()