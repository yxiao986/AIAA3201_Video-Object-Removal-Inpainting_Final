import os
import sys
import glob
import cv2
import numpy as np
import argparse
import shutil
import subprocess
import json

# Dynamically get absolute paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root_dir)

from huggingface_hub import hf_hub_download
from utils.diffusion_utils import run_sd_inpainting, get_auto_keyframe_indices
from utils.mask_utils import generate_random_stationary_mask, save_stationary_mask_sequence
from utils.metrics import evaluate_video_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: Generative Evaluation Pipeline")
    parser.add_argument("--dataset_name", type=str, required=True, help="e.g., tennis")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for Generative Models")
    
    parser.add_argument("--gt_data_dir", type=str, default=None, help="Path to original frames with dynamic objects")
    parser.add_argument("--gt_mask_dir", type=str, default=None, help="Path to dynamic GT masks")
    parser.add_argument("--clean_data_dir", type=str, default=None, help="Path to clean GT video frames for metrics")
    parser.add_argument("--n_keyframes", type=int, default=3, help="Number of SD keyframes (default: 3)")
    
    parser.add_argument("--method", type=str, choices=['baseline', 'sd2d', 'diffueraser'], default='diffueraser',
                        help="Choose the model to run: baseline (ProPainter), sd2d, or diffueraser.")
                        
    parser.add_argument("--output_base_dir", type=str, default=os.path.join(project_root_dir, "results", "part3_evaluation"))
    return parser.parse_args()

def run_propainter(data_dir, mask_dir, output_dir):
    propainter_dir = os.path.join(project_root_dir, "third_party", "ProPainter")
    propainter_script = os.path.join(propainter_dir, "inference_propainter.py")
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = project_root_dir + os.pathsep + custom_env.get("PYTHONPATH", "")
    try:
        subprocess.run([
            sys.executable, propainter_script,
            "--video", os.path.abspath(data_dir),
            "--mask", os.path.abspath(mask_dir),
            "--output", os.path.abspath(output_dir)
        ], check=True, cwd=propainter_dir, env=custom_env, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[Error] ProPainter execution failed: {e}")

def get_video_fps(video_path):
    if not os.path.exists(video_path): return 30.0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0

def generate_video_from_frames(frame_dir, output_path, target_fps=30.0):
    if os.path.exists(output_path):
        return True
    print(f"      -> [Auto-Fix] Compiling frames to video: {output_path} (FPS: {target_fps:.2f}) ...")
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.[pj][np][g]")))
    if not frames:
        print(f"[Error] Directory {frame_dir} is empty.")
        return False
        
    img = cv2.imread(frames[0])
    height, width, _ = img.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(target_fps), (width, height))
    for frame_path in frames:
        video.write(cv2.imread(frame_path))
    video.release()
    return True

def run_diffueraser_inference(data_dir, mask_dir, output_dir):
    diffueraser_dir = os.path.join(project_root_dir, "third_party", "DiffuEraser")
    if not os.path.isdir(diffueraser_dir):
        raise FileNotFoundError("DiffuEraser not found. Please clone it.")
    
    diffueraser_script = os.path.join(diffueraser_dir, "run_diffueraser.py") 
    clean_data_dir = os.path.normpath(data_dir)
    clean_mask_dir = os.path.normpath(mask_dir)
    input_video_mp4 = f"{clean_data_dir}.mp4"
    input_mask_mp4 = f"{clean_mask_dir}.mp4"
    
    if not generate_video_from_frames(clean_data_dir, input_video_mp4, target_fps=30.0): return
    real_fps = get_video_fps(input_video_mp4)
    if not generate_video_from_frames(clean_mask_dir, input_mask_mp4, target_fps=real_fps): return

    print("\n      -> [DiffuEraser] Starting native inference...")
    try:
        subprocess.run([
            sys.executable, diffueraser_script,
            "--input_video", os.path.abspath(input_video_mp4),
            "--input_mask", os.path.abspath(input_mask_mp4),
            "--save_path", os.path.abspath(output_dir)
        ], check=True, cwd=diffueraser_dir)
    except subprocess.CalledProcessError as e:
        print(f"[Error] DiffuEraser execution failed: {e}")

def load_frames_from_dir(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.[pj][np][g]")))
    return [cv2.imread(f) for f in files]

def extract_frames_from_propainter_output(output_dir, expected_count):
    mp4_files = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
    frames = []
    if mp4_files:
        cap = cv2.VideoCapture(mp4_files[0])
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
    else:
        img_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True))
        frames = [cv2.imread(f) for f in img_files]

    if len(frames) >= expected_count: return frames[:expected_count]
    elif len(frames) > 0: return frames
    else: return []

def run_pipeline(data_dir, mask_dir, output_dir, dataset_name, prompt, n_keyframes, method, is_stationary=False):
    img_files = sorted(glob.glob(os.path.join(data_dir, "*.[pj][np][g]")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    total_frames = len(img_files)

    method_out_dir = os.path.join(output_dir, method) 
    os.makedirs(method_out_dir, exist_ok=True)

    print(f"      -> Executing Method: [{method.upper()}]...")
    
    if method == 'baseline':
        run_propainter(data_dir, mask_dir, method_out_dir)
        
    elif method == 'sd2d':
        injected_data_dir = os.path.join(output_dir, "sd_injected_frames")
        injected_mask_dir = os.path.join(output_dir, "sd_injected_masks")
        os.makedirs(injected_data_dir, exist_ok=True)
        os.makedirs(injected_mask_dir, exist_ok=True)
        
        target_indices = get_auto_keyframe_indices(total_frames, n_keyframes)
        for idx, img_path in enumerate(img_files):
            filename = os.path.basename(img_path)
            img_out_path = os.path.join(injected_data_dir, filename)
            mask_filename = os.path.splitext(filename)[0] + '.png' if is_stationary else os.path.basename(mask_files[idx])
            mask_path = os.path.join(mask_dir, mask_filename)
            mask_out_path = os.path.join(injected_mask_dir, mask_filename)

            if idx in target_indices:
                run_sd_inpainting(img_path, mask_path, prompt, img_out_path)
                blank_mask = np.zeros_like(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
                cv2.imwrite(mask_out_path, blank_mask)
            else:
                shutil.copy(img_path, img_out_path)
                shutil.copy(mask_path, mask_out_path)
        
        run_propainter(injected_data_dir, injected_mask_dir, method_out_dir)
        
    elif method == 'diffueraser':
        run_diffueraser_inference(data_dir, mask_dir, method_out_dir)

    return method_out_dir, total_frames

def main():
    args = parse_args()
    print(f"\n{'='*70}\nPart 3: Generative Evaluation Pipeline [{args.dataset_name} | {args.method.upper()}]\n{'='*70}")
    
    method_base_dir = os.path.join(args.output_base_dir, args.dataset_name, args.method)
    
    if args.gt_data_dir and args.gt_mask_dir:
        print("\n>>> Starting Qualitative Evaluation (Dynamic Masking) <<<")
        qualitative_out_dir = os.path.join(method_base_dir, "qualitative")
        out_dir, _ = run_pipeline(
            args.gt_data_dir, args.gt_mask_dir, qualitative_out_dir, 
            args.dataset_name, args.prompt, args.n_keyframes, args.method, is_stationary=False
        )
        print(f" [Success] Visual results saved at: {out_dir}")
    
    if args.clean_data_dir:
        print("\n>>> Starting Quantitative Evaluation (Stationary Masking) <<<")
        quant_out_dir = os.path.join(method_base_dir, "quantitative")
        os.makedirs(quant_out_dir, exist_ok=True)
        
        clean_img_files = sorted(glob.glob(os.path.join(args.clean_data_dir, "*.[pj][np][g]")))
        sample_frame = cv2.imread(clean_img_files[0])
        stat_mask = generate_random_stationary_mask(sample_frame.shape[0], sample_frame.shape[1], 8, 45)
        
        stationary_mask_dir = os.path.join(quant_out_dir, "stationary_masks")
        save_stationary_mask_sequence(stat_mask, stationary_mask_dir, len(clean_img_files), [os.path.basename(f) for f in clean_img_files])

        out_dir, total_frames = run_pipeline(
            args.clean_data_dir, stationary_mask_dir, quant_out_dir, 
            args.dataset_name, args.prompt, args.n_keyframes, args.method, is_stationary=True
        )

        gt_frames = load_frames_from_dir(args.clean_data_dir)
        pred_frames = extract_frames_from_propainter_output(out_dir, total_frames)

        if len(gt_frames) == len(pred_frames) > 0:
            res = evaluate_video_quality(pred_frames, gt_frames)
            metrics = { f"Ours_{args.method.upper()}": res }
            
            print("\n" + "="*40)
            print(f" Quantitative Results ({args.method.upper()})")
            print("="*40)
            print(f"PSNR: {res['PSNR']:.2f}, SSIM: {res['SSIM']:.4f}")
            print("="*40)
            
            with open(os.path.join(quant_out_dir, "evaluation_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            print(f"[Error] Metric Calculation Failed due to frame count mismatch.")

if __name__ == '__main__':
    main()