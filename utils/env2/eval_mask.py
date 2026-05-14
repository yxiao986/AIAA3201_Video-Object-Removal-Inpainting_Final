import os
import cv2
import numpy as np
import argparse

def calculate_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) for two binary masks."""
    m1 = mask1 > 0
    m2 = mask2 > 0
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser(description="Evaluate binary mask quality with IoU mean (J_M) and IoU recall (J_R).")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted masks.")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground-truth masks.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to evaluate. Default: all matching frame ids.")
    parser.add_argument("--pred_pattern", default="dynamic_mask_{idx:04d}.png", help="Prediction filename pattern.")
    parser.add_argument("--gt_pattern", default="{idx:05d}.png", help="GT filename pattern.")
    parser.add_argument("--iou_recall_threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.num_frames is None:
        num_frames = len([f for f in os.listdir(args.gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    else:
        num_frames = args.num_frames
    ious = []
    
    print("Starting mask evaluation...")
    for i in range(num_frames):
        pred_filename = args.pred_pattern.format(idx=i)
        gt_filename = args.gt_pattern.format(idx=i)
        pred_path = os.path.join(args.pred_dir, pred_filename)
        gt_path = os.path.join(args.gt_dir, gt_filename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: prediction file not found: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: ground-truth file not found: {gt_path}")
            continue
            
        # Read masks as grayscale images so any nonzero value is foreground.
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Match the prediction size to the GT size before computing IoU.
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Binarize masks: any nonzero pixel is foreground.
        pred_bin = pred_img > 0
        gt_bin = gt_img > 0
        
        iou = calculate_iou(pred_bin, gt_bin)
        ious.append(iou)
        
    if not ious:
        print("No frame was evaluated successfully. Please check the input paths and filename patterns.")
        return
        
    j_m = np.mean(ious)
    j_r = np.mean(np.array(ious) > args.iou_recall_threshold)
    
    print("-" * 30)
    print(f"Evaluation over {len(ious)} frame(s):")
    print(f"Mask Quality - IoU mean (J_M): {j_m:.4f}")
    print(f"Mask Quality - IoU recall (J_R): {j_r:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
