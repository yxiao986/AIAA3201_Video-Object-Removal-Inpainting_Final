# Attempt to optimize Part 3; the quality difference is limited.
import os
import re
import shutil
import cv2
import numpy as np
import glob
import sys
import subprocess
from pathlib import Path
import inspect
import argparse
import torch

# ==========================================
# 1. Path configuration
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PARENT_DIR = PROJECT_ROOT / "data/scenes"

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs/part3/sam3_vggt4d_improve"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

VGGT4D_DIR = str(PROJECT_ROOT / "external/VGGT4D")
PROPAINTER_DIR = str(PROJECT_ROOT / "external/ProPainter")
VGGT_CKPT = str(PROJECT_ROOT / "external/VGGT4D/ckpts/model_tracker_fixed_e20.pt")

SAM3_MODEL_DIR = str(PROJECT_ROOT / "external/sam3_ms")
SAM3_CKPT_PATH = str(Path(SAM3_MODEL_DIR) / "sam3.pt")
SAM3_BPE_PATH = str(Path(SAM3_MODEL_DIR) / "bpe_simple_vocab_16e6.txt.gz")

SAM3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PYTHON_EXEC = "python"

CHUNK_SIZE = 20

RESULT_TXT_PATH = OUTPUT_BASE_DIR / "result.txt"
RESULT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

SCENES_TO_RUN = ["bmx-trees", "tennis"]

GT_MASK_DIRS = {
    "bmx-trees": str(PROJECT_ROOT / "data/gt_masks/bmx-trees"),
    "tennis": str(PROJECT_ROOT / "data/gt_masks/tennis"),
}

GT_VIDEO_PATHS = {
    "bmx-trees": str(PROJECT_ROOT / "data/prepared_eval/bmx-trees/gt_video.mp4"),
    "tennis": str(PROJECT_ROOT / "data/prepared_eval/tennis/gt_video.mp4"),
}

# ==========================================
# 2. Metric helpers
# ==========================================
def calculate_iou(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    
    # Empty foreground in both masks means a perfect static-background prediction.
    if union == 0:
        return 1.0
        
    return intersection / union

def evaluate_mask_quality(pred_masks_dir, gt_mask_dir, threshold=0.5):
    # Match frames by filename instead of relying on zip order.
    pred_mask_dict = {os.path.basename(p): p for p in glob.glob(os.path.join(pred_masks_dir, "*.png"))}
    gt_mask_dict = {os.path.basename(p): p for p in glob.glob(os.path.join(gt_mask_dir, "*.png"))}
    # Use only common frame names to avoid global shifts from missing boundary frames.
    common_files = sorted(list(set(pred_mask_dict.keys()) & set(gt_mask_dict.keys())))
    if not common_files:
        print("  [Warning] Predicted masks and GT masks have no common filenames. Check the directories.")
        return None, None
        
    ious = []
    for file_name in common_files:
        pred_path = pred_mask_dict[file_name]
        gt_path = gt_mask_dict[file_name] 
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  
        if pred_mask is None or gt_mask is None:
            continue      
        # Align GT resolution to the prediction.
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(calculate_iou(pred_mask, gt_mask))   
    if not ious:
        return None, None
    # Return raw ratios; format conversion can be handled by the report layer.
    j_m = np.mean(ious) 
    j_r = (np.sum(np.array(ious) > threshold) / len(ious)) 
    return j_m, j_r

def append_result_to_txt(result_txt_path, text):
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
 
 
def count_image_files(folder):
    if not os.path.exists(folder):
        return 0
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    total = 0
    for ext in exts:
        total += len(glob.glob(os.path.join(folder, ext)))
    return total
   
   
def parse_args():
    parser = argparse.ArgumentParser(description="Run the improved SAM3 + VGGT4D + ProPainter pipeline.")

    parser.add_argument("--input_dir", type=Path, default=INPUT_PARENT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_BASE_DIR)

    parser.add_argument(
        "--scene",
        type=str,
        default="both",
        choices=["bmx-trees", "tennis", "both"],
        help="Scene to run: bmx-trees, tennis, or both."
    )

    parser.add_argument("--vggt_ckpt", type=str, default=VGGT_CKPT)

    parser.add_argument("--sam3_ckpt", type=str, default=SAM3_CKPT_PATH)
    parser.add_argument("--sam3_bpe", type=str, default=SAM3_BPE_PATH)

    parser.add_argument("--gt_bmx", type=str, default=GT_MASK_DIRS["bmx-trees"])
    parser.add_argument("--gt_tennis", type=str, default=GT_MASK_DIRS["tennis"])
    parser.add_argument("--vggt4d_dir", type=str, default=VGGT4D_DIR,
                        help="Path to the local VGGT4D repository.")
    parser.add_argument("--propainter_dir", type=str, default=PROPAINTER_DIR,
                        help="Path to the local ProPainter repository.")
    parser.add_argument("--python_exec", type=str, default=PYTHON_EXEC,
                        help="Python executable used for external scripts.")
    parser.add_argument("--sam3_device", type=str, default=SAM3_DEVICE,
                        help="Device used by SAM3, e.g. cuda or cpu.")

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)

    return parser.parse_args()


def configure_from_args(args):
    global INPUT_PARENT_DIR, OUTPUT_BASE_DIR, RESULT_TXT_PATH
    global SAM3_CKPT_PATH, SAM3_BPE_PATH
    global SCENES_TO_RUN, GT_MASK_DIRS
    global CHUNK_SIZE
    global VGGT4D_DIR, PROPAINTER_DIR, VGGT_CKPT, PYTHON_EXEC, SAM3_DEVICE

    INPUT_PARENT_DIR = Path(args.input_dir)

    OUTPUT_BASE_DIR = Path(args.output_dir)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    RESULT_TXT_PATH = OUTPUT_BASE_DIR / "result.txt"
    RESULT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

    SAM3_CKPT_PATH = args.sam3_ckpt
    SAM3_BPE_PATH = args.sam3_bpe
    VGGT_CKPT = args.vggt_ckpt
    VGGT4D_DIR = args.vggt4d_dir
    PROPAINTER_DIR = args.propainter_dir
    PYTHON_EXEC = args.python_exec
    SAM3_DEVICE = args.sam3_device

    if args.scene == "both":
        SCENES_TO_RUN = ["bmx-trees", "tennis"]
    else:
        SCENES_TO_RUN = [args.scene]

    GT_MASK_DIRS = {
        "bmx-trees": args.gt_bmx,
        "tennis": args.gt_tennis,
    }

    CHUNK_SIZE = args.chunk_size

       
# ==========================================
# 3. Chunking helpers
# ==========================================
def prepare_chunked_dataset(orig_dir, chunked_parent_dir, scene_name, chunk_size=20):
    """
    Split a long sequence into shorter chunks to control VGGT4D attention memory.
    Frames are copied at original resolution.
    """
    img_paths = sorted(glob.glob(os.path.join(orig_dir, "*.[jp][pn]g")))
    if not img_paths: return
    
    for idx, p in enumerate(img_paths):
        # Compute the chunk index for the current frame.
        chunk_idx = idx // chunk_size
        # Build directory names such as bmx-trees_chunk_00.
        chunk_dir = chunked_parent_dir / f"{scene_name}_chunk_{chunk_idx:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files directly; this is faster and avoids re-encoding.
        shutil.copy(p, str(chunk_dir / os.path.basename(p)))
        
    print(f"  -> {scene_name} split into {len(img_paths)//chunk_size + 1} chunks ({chunk_size} frames per chunk, original resolution)")

# ==========================================
# merge_masks
# ==========================================
def merge_masks(orig_dir, mask_parent_dir, scene_name):
    """
    Merge chunk masks into a complete scene folder after VGGT4D.
    Reuse complete merged masks when they already exist.
    """
    final_scene_mask_dir = mask_parent_dir / scene_name
    final_scene_mask_dir.mkdir(parents=True, exist_ok=True)

    orig_images = sorted(glob.glob(os.path.join(orig_dir, "*.[jp][pn]g")))
    expected_num = len(orig_images)

    # Reuse a complete merged mask set when present.
    existing_num = count_image_files(final_scene_mask_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"  -> Scene {scene_name}: reusing complete merged masks ({existing_num}/{expected_num})")
        return final_scene_mask_dir

    chunk_dirs = sorted(mask_parent_dir.glob(f"{scene_name}_chunk_*"))

    all_mask_files = []
    for c_dir in chunk_dirs:
        chunk_masks = sorted((c_dir / "masks").glob("*.png"))
        all_mask_files.extend(chunk_masks)

    print(f"  -> Scene {scene_name}: {len(orig_images)} source frames, {len(all_mask_files)} masks found")

    for i, mask_file in enumerate(all_mask_files):
        if i >= len(orig_images):
            break

        orig_name = Path(orig_images[i]).stem
        new_mask_name = f"{orig_name}.png"

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, pure_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(str(final_scene_mask_dir / new_mask_name), pure_mask)

    # Keep cleanup for chunk-level output directories.
    for c_dir in chunk_dirs:
        shutil.rmtree(c_dir)

    return final_scene_mask_dir

# ==========================================
# mask_to_sam3
# ==========================================
def _mask_to_sam3_prompts(mask):
    """
    Build SAM3 box and foreground point prompts from a VGGT4D rough mask.
    This keeps the VGGT4D/ProPainter flow unchanged.
    """
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None, None, None

    ys, xs = np.where(fg > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Pad the box slightly so the prompt is not too tight.
    pad = 3
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)

    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    # Use the centroid of the largest connected component as the foreground point.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cx, cy = centroids[largest_idx]
    else:
        cx, cy = xs.mean(), ys.mean()

    point_coords = np.array([[float(cx), float(cy)]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    return box, point_coords, point_labels


def _mask_to_sam3_prompts(mask):
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None, None, None

    ys, xs = np.where(fg > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad = 3
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)

    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cx, cy = centroids[largest_idx]
    else:
        cx, cy = xs.mean(), ys.mean()

    point_coords = np.array([[float(cx), float(cy)]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    return box, point_coords, point_labels


def run_sam3_refine(image_dir, coarse_mask_dir, refined_mask_dir):
    print(f"\n[SAM3] Refining scene: {image_dir.name}")
    refined_mask_dir.mkdir(parents=True, exist_ok=True)

    expected_num = count_image_files(image_dir)
    existing_num = count_image_files(refined_mask_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"[SAM3] {image_dir.name} already has complete refined masks; reusing ({existing_num}/{expected_num})")
        return refined_mask_dir
    
    # The SAM3 path points to model files, not a source repository.
    if not os.path.exists(SAM3_CKPT_PATH):
        print(f"SAM3 checkpoint does not exist: {SAM3_CKPT_PATH}")
        shutil.copytree(coarse_mask_dir, refined_mask_dir, dirs_exist_ok=True)
        return refined_mask_dir

    if not os.path.exists(SAM3_BPE_PATH):
        print(f"SAM3 BPE does not exist: {SAM3_BPE_PATH}")
        shutil.copytree(coarse_mask_dir, refined_mask_dir, dirs_exist_ok=True)
        return refined_mask_dir

    try:
        # Use the sam3 package installed in the active environment.
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        # Use the image model with the interactive image predictor.
        sam_model = build_sam3_image_model(
            checkpoint_path=SAM3_CKPT_PATH,
            bpe_path=SAM3_BPE_PATH,
            load_from_HF=False,
            eval_mode=True,
            device=SAM3_DEVICE,
        )
        predictor = SAM3InteractiveImagePredictor(sam_model)

    except Exception as e:
        print(f"SAM3 initialization failed; falling back to VGGT4D coarse masks. Error: {e}")
        shutil.copytree(coarse_mask_dir, refined_mask_dir, dirs_exist_ok=True)
        return refined_mask_dir

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")))
    coarse_mask_paths = sorted(glob.glob(os.path.join(coarse_mask_dir, "*.png")))
    coarse_mask_map = {Path(p).stem: p for p in coarse_mask_paths}

    for img_path in image_paths:
        stem = Path(img_path).stem
        out_path = refined_mask_dir / f"{stem}.png"

        coarse_path = coarse_mask_map.get(stem, None)
        if coarse_path is None:
            continue

        image_bgr = cv2.imread(img_path)
        coarse_mask = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or coarse_mask is None:
            continue

        if coarse_mask.shape[:2] != image_bgr.shape[:2]:
            coarse_mask = cv2.resize(
                coarse_mask,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        if np.count_nonzero(coarse_mask) == 0:
            cv2.imwrite(str(out_path), coarse_mask)
            continue

        box, point_coords, point_labels = _mask_to_sam3_prompts(coarse_mask)
        if box is None:
            cv2.imwrite(str(out_path), coarse_mask)
            continue

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # set_image computes image embeddings before prediction.
            predictor.set_image(image_rgb)

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False
            )

            if masks is None or len(masks) == 0:
                refined_mask = coarse_mask
            else:
                refined_mask = (masks[0].astype(np.uint8) * 255)

                # Fall back to the coarse mask if SAM3 returns an empty or tiny mask.
                if refined_mask.sum() == 0 or refined_mask.sum() < 0.1 * coarse_mask.sum():
                    refined_mask = coarse_mask

            cv2.imwrite(str(out_path), refined_mask)

        except Exception as e:
            print(f"[SAM3] {stem} refinement failed; falling back to coarse mask. Error: {e}")
            cv2.imwrite(str(out_path), coarse_mask)

        finally:
            # Different SAM3 versions may expose different reset methods.
            try:
                predictor.reset_predictor()
            except Exception:
                try:
                    predictor.reset_image()
                except Exception:
                    pass

    return refined_mask_dir

# ==========================================
# 4. External model calls
# ==========================================
def run_vggt4d(parent_input_dir, parent_output_dir):
    print(f"\n[VGGT4D] Processing chunked dataset: {parent_input_dir}")
    parent_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXEC, f"{VGGT4D_DIR}/demo_vggt4dzuri.py",
        "--input_dir", str(parent_input_dir),
        "--output_dir", str(parent_output_dir),
        "--vggt_ckpt", str(VGGT_CKPT),
    ]
    subprocess.run(cmd, check=True, cwd=VGGT4D_DIR)

def run_propainter(dataset_path, mask_dir, output_dir):
    print(f"\n[ProPainter] Inpainting full sequence: {dataset_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_num = count_image_files(dataset_path)
    existing_num = count_image_files(output_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"[ProPainter] {dataset_path.name} already has complete output; reusing ({existing_num}/{expected_num})")
        return

    cmd = [
        PYTHON_EXEC, f"{PROPAINTER_DIR}/inference_propainter.py",
        "--video", str(dataset_path),
        "--mask", str(mask_dir),
        "--output", str(output_dir)
    ]
    subprocess.run(cmd, check=True, cwd=PROPAINTER_DIR)

# ==========================================
# 5. Main batch pipeline
# ==========================================
def main_pipeline():
    print(f"{'='*50}\nStarting global chunked batch processing (original resolution)\nInput parent directory: {INPUT_PARENT_DIR}\n{'='*50}")
    append_result_to_txt(RESULT_TXT_PATH, "\n" + "=" * 80)
    append_result_to_txt(RESULT_TXT_PATH, "Starting a new evaluation run")
    scene_dirs = [
        d for d in INPUT_PARENT_DIR.iterdir()
        if d.is_dir() and d.name in SCENES_TO_RUN
    ]
    if not scene_dirs: return

    chunked_parent_dir = OUTPUT_BASE_DIR / "chunked_input"
    mask_parent_dir = OUTPUT_BASE_DIR / "vggt4d_masks"

    # Run VGGT4D only for scenes without complete coarse merged masks.
    scenes_need_vggt4d = []

    print("\n>>> Stage 1: checking whether chunks and VGGT4D are needed")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        coarse_merged_dir = mask_parent_dir / scene_name

        expected_num = count_image_files(scene_dir)
        existing_num = count_image_files(coarse_merged_dir)

        if expected_num > 0 and existing_num == expected_num:
            print(f"  -> {scene_name}: complete coarse masks already exist; skipping chunking and VGGT4D ({existing_num}/{expected_num})")
        else:
            print(f"  -> {scene_name}: coarse masks are incomplete; preparing to rerun VGGT4D")
            prepare_chunked_dataset(scene_dir, chunked_parent_dir, scene_name, chunk_size=CHUNK_SIZE)
            scenes_need_vggt4d.append(scene_dir)

    # Run VGGT4D only when some scenes are missing coarse masks.
    if scenes_need_vggt4d:
        try:
            run_vggt4d(chunked_parent_dir, mask_parent_dir)
        except subprocess.CalledProcessError as e:
            print(f"VGGT4D failed: {e}")
            return
        finally:
            if chunked_parent_dir.exists():
                shutil.rmtree(chunked_parent_dir)
    else:
        print("\n>>> Stage 2: all scenes already have coarse masks; skipping VGGT4D")

    print("\n>>> Stage 3: mask merge -> SAM3 -> ProPainter")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        inpainting_out_dir = OUTPUT_BASE_DIR / scene_name / "inpainted_results"
        sam3_refined_mask_dir = OUTPUT_BASE_DIR / "sam3_refined_masks" / scene_name

        coarse_merged_dir = mask_parent_dir / scene_name
        expected_num = count_image_files(scene_dir)
        coarse_existing_num = count_image_files(coarse_merged_dir)

        # Reuse complete coarse merged masks; otherwise merge this VGGT4D run.
        if expected_num > 0 and coarse_existing_num == expected_num:
            print(f"  -> {scene_name}: reusing existing coarse merged masks ({coarse_existing_num}/{expected_num})")
            final_scene_mask_dir = coarse_merged_dir
        else:
            final_scene_mask_dir = merge_masks(scene_dir, mask_parent_dir, scene_name)

        # Refine coarse masks with SAM3; the function reuses complete outputs.
        final_scene_mask_dir = run_sam3_refine(
            image_dir=scene_dir,
            coarse_mask_dir=final_scene_mask_dir,
            refined_mask_dir=sam3_refined_mask_dir
        )
           
        # 2. Evaluate masks.
        j_m, j_r = None, None
        gt_mask_dir = GT_MASK_DIRS.get(scene_name, None)
        if gt_mask_dir and os.path.exists(gt_mask_dir):
            j_m, j_r = evaluate_mask_quality(final_scene_mask_dir, gt_mask_dir)
            if j_m is not None:
                print(f"[{scene_name}] Mask Quality -> J_M: {j_m:.4f}, J_R: {j_r:.4f}")
        else:
            print(f"[{scene_name}] GT mask path not found: {gt_mask_dir}")

        # 3. Run ProPainter.
        try:
            run_propainter(scene_dir, final_scene_mask_dir, inpainting_out_dir)
        except subprocess.CalledProcessError as e:
            print(f"ProPainter failed for {scene_name}: {e}")
            continue

        # 5. Append results to result.txt.
        result_line = (
            f"[{scene_name}] "
            f"J_M: {j_m:.4f} | J_R: {j_r:.4f} | "
            if (j_m is not None and j_r is not None)
            else f"[{scene_name}] some metrics failed | "
                 f"J_M={j_m}, J_R={j_r}"
        )
        print(result_line)
        append_result_to_txt(RESULT_TXT_PATH, result_line)
        
    print(f"\nAll processing complete. Results saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    args = parse_args()
    configure_from_args(args)
    main_pipeline()
