import os
import sys
import glob
import cv2
import json
import argparse
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, "..", "..", ".."))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from part3.env1.ProPainter_Explore import main as propainter_explore
from utils.mask_utils import generate_random_stationary_mask, save_stationary_mask_sequence
from utils.metrics import evaluate_video_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: Quantitative Evaluation on DAVIS (Stationary Mask)")
    parser.add_argument("--davis_root", type=str, default=os.path.join(project_root_dir, "data", "DAVIS"), 
                        help="Path to the DAVIS root directory")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(project_root_dir, "results", "part3", "ProPainter_Explore", "DAVIS"), 
                        help="Directory to save quantitative results")
    parser.add_argument("--method", type=str, choices=['baseline', 'diffueraser', 'sd2d'], required=True,
                        help="Which method to run")
    parser.add_argument("--target_seqs", type=str, nargs="+", default=[],
                        help="Specific sequences to evaluate. If empty, runs ALL available sequences.")
    parser.add_argument("--propainter_dir", type=str,
                        default=os.path.join(project_root_dir, "third_party", "ProPainter"),
                        help="Path to the local ProPainter repository.")
    parser.add_argument("--diffueraser_dir", type=str,
                        default=os.path.join(project_root_dir, "third_party", "DiffuEraser"),
                        help="Path to the local DiffuEraser repository.")
    parser.add_argument("--python_exec", type=str, default=sys.executable,
                        help="Python executable used for external inference scripts.")
    return parser.parse_args()

def main():
    args = parse_args()
    propainter_explore.configure_external_tools(args)
    jpeg_dir = os.path.join(args.davis_root, "JPEGImages", "480p")
    
    if not os.path.exists(jpeg_dir):
        print(f"[Error] DAVIS directory not found at {jpeg_dir}")
        return

    sequences = args.target_seqs if args.target_seqs else sorted([d for d in os.listdir(jpeg_dir) if os.path.isdir(os.path.join(jpeg_dir, d))])
        
    print(f"\n{'='*70}")
    print(f" Part 3: DAVIS Quantitative Evaluation (Stationary Masking)")
    print(f" Method: {args.method.upper()}")
    print(f" Sequences to process: {len(sequences)}")
    print(f"{'='*70}\n")
    
    global_metrics = {}
    all_psnr = []
    all_ssim = []
    
    for seq in sequences:
        print(f"\n>> Processing Sequence: [{seq}] for Quantitative Metrics")
        clean_data_dir = os.path.join(jpeg_dir, seq)
        
        quant_out_dir = os.path.join(args.output_dir, seq, args.method, "quantitative")
        os.makedirs(quant_out_dir, exist_ok=True)
        
        clean_img_files = sorted(glob.glob(os.path.join(clean_data_dir, "*.[pj][np][g]")))
        if not clean_img_files:
            print(f"   [Warning] No frames found in {clean_data_dir}. Skipping.")
            continue
            
        stationary_mask_dir = os.path.join(quant_out_dir, "stationary_masks")
        sample_frame = cv2.imread(clean_img_files[0])
        stat_mask = generate_random_stationary_mask(sample_frame.shape[0], sample_frame.shape[1], 8, 45)
        save_stationary_mask_sequence(stat_mask, stationary_mask_dir, len(clean_img_files), [os.path.basename(f) for f in clean_img_files])

        out_dir, total_frames = propainter_explore.run_pipeline(
            data_dir=clean_data_dir, 
            mask_dir=stationary_mask_dir, 
            output_dir=quant_out_dir, 
            dataset_name=f"DAVIS_{seq}", 
            prompt="",  
            n_keyframes=3, 
            method=args.method, 
            is_stationary=True
        )

        gt_frames = [cv2.imread(f) for f in clean_img_files]
        pred_frames = propainter_explore.extract_frames_from_propainter_output(out_dir, total_frames)

        if len(gt_frames) == len(pred_frames) > 0:
            gt_h, gt_w = gt_frames[0].shape[:2]
            aligned_pred_frames = []
            for pf in pred_frames:
                if pf.shape[:2] != (gt_h, gt_w):
                    pf = cv2.resize(pf, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
                aligned_pred_frames.append(pf)

            res = evaluate_video_quality(aligned_pred_frames, gt_frames)
            psnr, ssim = res['PSNR'], res['SSIM']
            print(f"   [Result] {seq} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
            
            metrics = { f"Ours_{args.method.upper()}": res }
            with open(os.path.join(quant_out_dir, "evaluation_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            global_metrics[seq] = res
        else:
            print(f"   [Error] Frame mismatch for {seq}. GT: {len(gt_frames)}, Pred: {len(pred_frames)}")

    if all_psnr:
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        print(f"\n{'='*70}")
        print(f" GLOBAL QUANTITATIVE RESULTS ({args.method.upper()})")
        print(f"{'='*70}")
        print(f" Average PSNR: {avg_psnr:.2f}")
        print(f" Average SSIM: {avg_ssim:.4f}")
        print(f"{'='*70}")
        
        global_metrics["AVERAGE_DATASET_SCORE"] = {"PSNR": avg_psnr, "SSIM": avg_ssim}
        global_json_path = os.path.join(args.output_dir, f"davis_global_quant_{args.method}.json")
        with open(global_json_path, "w") as f:
            json.dump(global_metrics, f, indent=4)

if __name__ == '__main__':
    main()
