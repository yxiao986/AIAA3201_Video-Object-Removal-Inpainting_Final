# Process the BMX-Trees and Tennis scenes.
import os
import sys
import math
import shutil
import cv2
import numpy as np
import glob
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import re

# ==========================================
# 1. Path configuration
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VGGT_CKPT = str(PROJECT_ROOT / "external/VGGT4D/ckpts/model_tracker_fixed_e20.pt")

INPUT_PARENT_DIR = PROJECT_ROOT / "data/scenes"

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs/part3/sam3_vggt4d"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_TXT_PATH = OUTPUT_BASE_DIR / "result.txt"

MASK_GT_DIRS = {
    "bmx-trees": str(PROJECT_ROOT / "data/gt_masks/bmx-trees"),
    "tennis": str(PROJECT_ROOT / "data/gt_masks/tennis"),
}

VGGT4D_DIR = str(PROJECT_ROOT / "external/VGGT4D")
PROPAINTER_DIR = str(PROJECT_ROOT / "external/ProPainter")

SAM3_DIR = str(PROJECT_ROOT / "external/sam3")
SAM3_CHECKPOINT = str(PROJECT_ROOT / "external/sam3_ms/sam3.pt")
SAM3_BPE_PATH = str(PROJECT_ROOT / "external/sam3_ms/bpe_simple_vocab_16e6.txt.gz")
SAM3_DEVICE = "cuda"

PYTHON_EXEC = "python"

ENABLE_SAM3_REFINE = True
CHUNK_SIZE = 20

SAM3_KEEP_LARGEST_COMPONENT = True
SAM3_MIN_IOU_WITH_ROUGH_MASK = 0.05

# ==========================================
# 2. Metric helpers
# ==========================================
def calculate_iou(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return intersection / union if union > 0 else 1.0

def binarize_mask(mask):
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = np.asarray(mask)

    # Support 0/1 masks, 0/255 masks, and grayscale probability maps.
    if mask.max() <= 1:
        return ((mask > 0).astype(np.uint8) * 255)
    else:
        return ((mask > 127).astype(np.uint8) * 255)


def choose_best_polarity(pred_paths, gt_paths, sample_k=10):
    """
    Used for Tennis only.
    Probe four polarity combinations on the first sample_k frames and keep the
    one with the highest mean IoU.
    Returns: pred_invert, gt_invert.
    """
    pair_num = min(len(pred_paths), len(gt_paths), sample_k)
    if pair_num == 0:
        return False, False

    modes = [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ]

    best_mode = (False, False)
    best_score = -1.0

    for pred_inv, gt_inv in modes:
        vals = []
        for pred_path, gt_path in zip(pred_paths[:pair_num], gt_paths[:pair_num]):
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if pred_mask is None or gt_mask is None:
                continue

            pred_mask = binarize_mask(pred_mask)
            gt_mask = binarize_mask(gt_mask)
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            if pred_inv:
                pred_mask = 255 - pred_mask
            if gt_inv:
                gt_mask = 255 - gt_mask

            vals.append(calculate_iou(pred_mask, gt_mask))

        score = float(np.mean(vals)) if vals else -1.0
        if score > best_score:
            best_score = score
            best_mode = (pred_inv, gt_inv)

    print(f"[tennis] Auto-selected polarity: pred_invert={best_mode[0]}, gt_invert={best_mode[1]}, probe_mean_iou={best_score:.4f}")
    return best_mode

def natural_key(x):
    x = str(x)
    name = os.path.basename(x)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def natural_sorted_glob(pattern):
    return sorted(glob.glob(pattern), key=natural_key)

def evaluate_mask_quality(pred_masks_dir, gt_mask_dir, threshold=0.5, scene_name=None):
    pred_mask_paths = sorted(glob.glob(os.path.join(pred_masks_dir, "*.png")))
    gt_mask_paths = sorted(glob.glob(os.path.join(gt_mask_dir, "*.png")))

    if not pred_mask_paths or not gt_mask_paths:
        print(f"[evaluate_mask_quality] Empty directory: pred={pred_masks_dir}, gt={gt_mask_dir}")
        return None, None

    pair_num = min(len(pred_mask_paths), len(gt_mask_paths))
    if len(pred_mask_paths) != len(gt_mask_paths):
        print(
            f"[evaluate_mask_quality] Frame counts differ; evaluating the first {pair_num} pairs: "
            f"pred={len(pred_mask_paths)}, gt={len(gt_mask_paths)}"
        )

    pred_invert, gt_invert = False, False

    # Only Tennis needs automatic polarity probing; keep the BMX flow unchanged.
    if scene_name == "tennis":
        pred_invert, gt_invert = choose_best_polarity(
            pred_mask_paths[:pair_num],
            gt_mask_paths[:pair_num],
            sample_k=10
        )

    ious = []
    for pred_path, gt_path in zip(pred_mask_paths[:pair_num], gt_mask_paths[:pair_num]):
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None or gt_mask is None:
            continue

        pred_mask = binarize_mask(pred_mask)
        gt_mask = binarize_mask(gt_mask)
        gt_mask = cv2.resize(
            gt_mask,
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        if pred_invert:
            pred_mask = 255 - pred_mask
        if gt_invert:
            gt_mask = 255 - gt_mask

        ious.append(calculate_iou(pred_mask, gt_mask))

    if not ious:
        return None, None

    ious = np.array(ious, dtype=np.float32)
    return float(np.mean(ious)), float(np.mean(ious > threshold))


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT4D coarse masks, optional SAM3 refinement, and ProPainter.")

    parser.add_argument("--input_dir", type=Path, default=INPUT_PARENT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_BASE_DIR)

    parser.add_argument(
        "--scene",
        type=str,
        default="both",
        choices=["bmx-trees", "tennis", "both"],
        help="Scene to run: bmx-trees, tennis, or both."
    )

    parser.add_argument("--vggt_ckpt", type=str, default=str(PROJECT_ROOT / "external/VGGT4D/ckpts/model_tracker_fixed_e20.pt"))

    parser.add_argument("--sam3_ckpt", type=str, default=SAM3_CHECKPOINT)
    parser.add_argument("--sam3_bpe", type=str, default=SAM3_BPE_PATH)

    parser.add_argument("--gt_bmx", type=str, default=MASK_GT_DIRS["bmx-trees"])
    parser.add_argument("--gt_tennis", type=str, default=MASK_GT_DIRS["tennis"])
    parser.add_argument("--vggt4d_dir", type=str, default=VGGT4D_DIR,
                        help="Path to the local VGGT4D repository.")
    parser.add_argument("--propainter_dir", type=str, default=PROPAINTER_DIR,
                        help="Path to the local ProPainter repository.")
    parser.add_argument("--python_exec", type=str, default=PYTHON_EXEC,
                        help="Python executable used for external scripts.")
    parser.add_argument("--sam3_device", type=str, default=SAM3_DEVICE,
                        help="Device used by SAM3, e.g. cuda or cpu.")

    parser.add_argument("--disable_sam3", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)

    return parser.parse_args()

def configure_from_args(args):
    global INPUT_PARENT_DIR, OUTPUT_BASE_DIR, RESULT_TXT_PATH
    global MASK_GT_DIRS
    global SAM3_CHECKPOINT, SAM3_BPE_PATH
    global ENABLE_SAM3_REFINE, CHUNK_SIZE
    global VGGT_CKPT
    global VGGT4D_DIR, PROPAINTER_DIR, PYTHON_EXEC, SAM3_DEVICE
    VGGT_CKPT = args.vggt_ckpt
    
    INPUT_PARENT_DIR = Path(args.input_dir)

    OUTPUT_BASE_DIR = Path(args.output_dir)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TXT_PATH = OUTPUT_BASE_DIR / "result.txt"

    MASK_GT_DIRS = {
        "bmx-trees": args.gt_bmx,
        "tennis": args.gt_tennis,
    }

    SAM3_CHECKPOINT = args.sam3_ckpt
    SAM3_BPE_PATH = args.sam3_bpe
    VGGT4D_DIR = args.vggt4d_dir
    PROPAINTER_DIR = args.propainter_dir
    PYTHON_EXEC = args.python_exec
    SAM3_DEVICE = args.sam3_device

    ENABLE_SAM3_REFINE = not args.disable_sam3
    CHUNK_SIZE = args.chunk_size
    
def append_result_line(result_txt_path, line):
    result_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        
# ==========================================
# 3. Chunking helpers
# ==========================================
def prepare_chunked_dataset(orig_dir, chunked_parent_dir, scene_name, chunk_size=20):
    """
    Split a long video sequence into chunks to control VGGT4D memory usage.
    """
    img_paths = natural_sorted_glob(os.path.join(orig_dir, "*.[jp][pn]g"))
    if not img_paths:
        return

    num_chunks = math.ceil(len(img_paths) / chunk_size)

    for idx, p in enumerate(img_paths):
        chunk_idx = idx // chunk_size
        chunk_dir = chunked_parent_dir / f"{scene_name}_chunk_{chunk_idx:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, str(chunk_dir / os.path.basename(p)))

    print(f"  -> {scene_name} split into {num_chunks} chunks ({chunk_size} frames per chunk, original resolution)")


def merge_masks(orig_dir, mask_parent_dir, scene_name):
    """
    Merge chunk-level masks back into a full scene in the original frame order.
    """
    final_scene_mask_dir = mask_parent_dir / scene_name
    if final_scene_mask_dir.exists():
        shutil.rmtree(final_scene_mask_dir)
    final_scene_mask_dir.mkdir(parents=True, exist_ok=True)

    orig_images = natural_sorted_glob(os.path.join(orig_dir, "*.[jp][pn]g"))
    chunk_dirs = sorted(mask_parent_dir.glob(f"{scene_name}_chunk_*"), key=lambda p: natural_key(p.name))

    all_mask_files = []
    for c_dir in chunk_dirs:
        chunk_masks = sorted((c_dir / "masks").glob("*.png"), key=lambda p: natural_key(p.name))
        all_mask_files.extend(chunk_masks)

    print(f"  -> Scene {scene_name}: {len(orig_images)} source frames, {len(all_mask_files)} masks to merge")

    for i, mask_file in enumerate(all_mask_files):
        if i >= len(orig_images):
            break

        orig_name = Path(orig_images[i]).stem
        new_mask_name = f"{orig_name}.png"

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, pure_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(final_scene_mask_dir / new_mask_name), pure_mask)

    return final_scene_mask_dir


# ==========================================
# 4. SAM3 refinement helpers.
#    VGGT4D produces rough masks; the rough masks are converted to box prompts,
#    refined frame-by-frame by SAM3, and then returned to the temporal pipeline.
# ==========================================
def mask_to_xywh_norm(mask):
    """
    Convert a rough mask to the normalized SAM3 xywh box format:
    [center_x, center_y, width, height], all in [0, 1].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    bw = max((x2 - x1 + 1) / w, 1.0 / w)
    bh = max((y2 - y1 + 1) / h, 1.0 / h)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h

    return [float(cx), float(cy), float(bw), float(bh)]


def keep_largest_component(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    kept = np.zeros_like(binary_mask)
    kept[labels == largest_label] = 255
    return kept


def postprocess_mask(mask, target_shape):
    """
    Convert a model output to a uint8 binary mask and apply light denoising.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    # Outputs may be bools, probabilities, or logits; normalize them to binary masks.
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255
    else:
        if mask.max() <= 1.0:
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255

    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if SAM3_KEEP_LARGEST_COMPONENT:
        mask = keep_largest_component(mask)

    return mask


def build_sam3_processor():
    """
    Initialize SAM3 from the installed Python package and local checkpoint/BPE.
    This does not depend on the local source tree layout.
    """
    import torch
    from PIL import Image  # noqa: F401

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        raise RuntimeError(
            "The SAM3 Python package is not installed correctly. Run: pip install -U sam3\n"
            f"Original error: {e}"
        ) from e

    device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"
    build_kwargs = {"device": device}

    if os.path.exists(SAM3_CHECKPOINT):
        build_kwargs["checkpoint_path"] = SAM3_CHECKPOINT
    else:
        raise FileNotFoundError(f"SAM3_CHECKPOINT does not exist: {SAM3_CHECKPOINT}")

    if os.path.exists(SAM3_BPE_PATH):
        build_kwargs["bpe_path"] = SAM3_BPE_PATH
    else:
        raise FileNotFoundError(f"SAM3_BPE_PATH does not exist: {SAM3_BPE_PATH}")

    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(model)
    return processor, device

def refine_one_mask_with_sam3(processor, image_bgr, rough_mask):
    """
    Build a box prompt from the rough mask, refine it with SAM3, and choose the
    candidate with the highest IoU against the rough mask.
    """
    from PIL import Image

    prompt_box = mask_to_xywh_norm(rough_mask)
    if prompt_box is None:
        return rough_mask

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    state = processor.set_image(image_pil)
    output = processor.add_geometric_prompt(box=prompt_box, label=True, state=state)

    candidate_masks = output.get("masks", None)
    if candidate_masks is None:
        return rough_mask

    if hasattr(candidate_masks, "detach"):
        candidate_masks = candidate_masks.detach().cpu().numpy()

    candidate_masks = np.asarray(candidate_masks)

    # Common layouts: [N, H, W] or [1, N, H, W].
    if candidate_masks.ndim == 4:
        candidate_masks = candidate_masks[0]

    if candidate_masks.ndim == 2:
        candidate_masks = candidate_masks[None, ...]

    best_mask = None
    best_iou = -1.0

    for cand in candidate_masks:
        try:
            cand_bin = postprocess_mask(cand, rough_mask.shape)
        except Exception:
            continue

        iou = calculate_iou(cand_bin, rough_mask)
        if iou > best_iou:
            best_iou = iou
            best_mask = cand_bin

    if best_mask is None or best_iou < SAM3_MIN_IOU_WITH_ROUGH_MASK:
        return rough_mask

    return best_mask


def refine_chunk_masks_with_sam3(chunked_parent_dir, raw_mask_parent_dir, refined_mask_parent_dir, scene_name):
    """
    Refine all chunk masks for one scene with SAM3.
    Inputs:
      - chunked_parent_dir: source chunk image directory.
      - raw_mask_parent_dir: VGGT4D raw output directory.
      - refined_mask_parent_dir: SAM3-refined output directory.
    """
    import torch

    print(f"\n[SAM3] Refining scene: {scene_name}")
    processor, device = build_sam3_processor()

    chunk_input_dirs = sorted(chunked_parent_dir.glob(f"{scene_name}_chunk_*"))
    total_frames = 0

    for chunk_input_dir in chunk_input_dirs:
        raw_mask_dir = raw_mask_parent_dir / chunk_input_dir.name / "masks"
        refined_mask_dir = refined_mask_parent_dir / chunk_input_dir.name / "masks"
        refined_mask_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(glob.glob(os.path.join(chunk_input_dir, "*.[jp][pn]g")))
        rough_mask_paths = sorted(raw_mask_dir.glob("*.png"))

        if not frame_paths or not rough_mask_paths:
            print(f"  -> Skipping {chunk_input_dir.name}: missing frames or rough masks")
            continue

        pair_num = min(len(frame_paths), len(rough_mask_paths))
        if len(frame_paths) != len(rough_mask_paths):
            print(
                f"  -> Warning {chunk_input_dir.name}: frame={len(frame_paths)} / mask={len(rough_mask_paths)}; "
                f"processing the first {pair_num} pairs"
            )

        for frame_path, rough_mask_path in zip(frame_paths[:pair_num], rough_mask_paths[:pair_num]):
            image = cv2.imread(frame_path)
            rough_mask = cv2.imread(str(rough_mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or rough_mask is None:
                continue

            refined_mask = refine_one_mask_with_sam3(processor, image, rough_mask)

            save_name = Path(frame_path).stem + ".png"
            cv2.imwrite(str(refined_mask_dir / save_name), refined_mask)
            total_frames += 1

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[SAM3] Finished scene {scene_name}; processed {total_frames} frames")


# ==========================================
# 5. External model calls
# ==========================================
def run_vggt4d(parent_input_dir, parent_output_dir):
    print(f"\n[VGGT4D] Processing chunked dataset: {parent_input_dir}")
    parent_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXEC,
        f"{VGGT4D_DIR}/demo_vggt4dzuri.py",
        "--input_dir", str(parent_input_dir),
        "--output_dir", str(parent_output_dir),
        "--vggt_ckpt", str(VGGT_CKPT),
    ]
    subprocess.run(cmd, check=True, cwd=VGGT4D_DIR)


def run_propainter(dataset_path, mask_dir, output_dir):
    print(f"\n[ProPainter] Inpainting full sequence: {dataset_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXEC,
        f"{PROPAINTER_DIR}/inference_propainter.py",
        "--video", str(dataset_path),
        "--mask", str(mask_dir),
        "--output", str(output_dir)
    ]
    subprocess.run(cmd, check=True, cwd=PROPAINTER_DIR)


# ==========================================
# 6. Main pipeline
# ==========================================
def main_pipeline(selected_scene):
    print(
        f"{'=' * 60}\n"
        f"Starting pipeline: scene data + VGGT4D coarse masks + SAM3 refinement + ProPainter\n"
        f"Input directory: {INPUT_PARENT_DIR}\n"
        f"Output directory: {OUTPUT_BASE_DIR}\n"
        f"{'=' * 60}"
    )

    all_scene_dirs = {d.name: d for d in INPUT_PARENT_DIR.iterdir() if d.is_dir()}

    if selected_scene == "both":
        wanted_names = ["bmx-trees", "tennis"]
    else:
        wanted_names = [selected_scene]

    scene_dirs = [all_scene_dirs[name] for name in wanted_names if name in all_scene_dirs]

    if not scene_dirs:
        print("No processable scene directory found; exiting.")
        return

    chunked_parent_dir = OUTPUT_BASE_DIR / "chunked_input"
    raw_mask_parent_dir = OUTPUT_BASE_DIR / "vggt4d_masks_raw"
    refined_mask_parent_dir = OUTPUT_BASE_DIR / "sam3_refined_masks_chunks"
    merged_mask_parent_dir = OUTPUT_BASE_DIR / "merged_masks"

    # --- Stage 1: temporal chunking ---
    print("\n>>> Stage 1: preparing chunked dataset")
    for scene_dir in scene_dirs:
        prepare_chunked_dataset(scene_dir, chunked_parent_dir, scene_dir.name, chunk_size=CHUNK_SIZE)

    # --- Stage 2: VGGT4D coarse masks ---
    print("\n>>> Stage 2: running VGGT4D for coarse masks")
    try:
        run_vggt4d(chunked_parent_dir, raw_mask_parent_dir)
    except subprocess.CalledProcessError as e:
        print(f"VGGT4D failed: {e}")
        return

    # --- Stage 3: SAM3 refinement ---
    active_mask_parent_dir = raw_mask_parent_dir
    if ENABLE_SAM3_REFINE:
        print("\n>>> Stage 3: refining VGGT4D rough masks with SAM3")
        try:
            for scene_dir in scene_dirs:
                refine_chunk_masks_with_sam3(
                    chunked_parent_dir=chunked_parent_dir,
                    raw_mask_parent_dir=raw_mask_parent_dir,
                    refined_mask_parent_dir=refined_mask_parent_dir,
                    scene_name=scene_dir.name
                )
            active_mask_parent_dir = refined_mask_parent_dir
        except Exception as e:
            print(f"SAM3 refinement failed; falling back to VGGT4D rough masks. Error: {e}")
            active_mask_parent_dir = raw_mask_parent_dir

    # Chunk images are only needed during VGGT4D/SAM3 processing.
    if chunked_parent_dir.exists():
        shutil.rmtree(chunked_parent_dir)

    run_summary_parts = [f"time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"scene={selected_scene}"]
    # --- Stage 4: merge masks and run ProPainter ---
    print("\n>>> Stage 4: merging masks and running video inpainting")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_result_parts = [f"{scene_name}"]
        scene_output_dir = OUTPUT_BASE_DIR / scene_name
        inpainting_out_dir = scene_output_dir / "inpainted_results"

        # 4.1 Merge full-sequence masks.
        final_scene_mask_dir = merge_masks(
            orig_dir=scene_dir,
            mask_parent_dir=active_mask_parent_dir,
            scene_name=scene_name
        )

        # Keep an additional copy under merged_masks for inspection.
        merged_scene_dir = merged_mask_parent_dir / scene_name
        if merged_scene_dir.exists():
            shutil.rmtree(merged_scene_dir)
        shutil.copytree(final_scene_mask_dir, merged_scene_dir)

        # 4.2 Evaluate masks.
        j_m, j_r = None, None
        gt_mask_dir = MASK_GT_DIRS.get(scene_name, None)

        if gt_mask_dir is not None and os.path.exists(gt_mask_dir):
            j_m, j_r = evaluate_mask_quality(final_scene_mask_dir, gt_mask_dir, scene_name=scene_name)
            if j_m is not None:
                print(f"[{scene_name}] Mask Quality -> J_M: {j_m:.4f}, J_R: {j_r:.4f}")
                scene_result_parts.append(f"JM={j_m:.4f}")
                scene_result_parts.append(f"JR={j_r:.4f}")
            else:
                scene_result_parts.append("JM=None")
                scene_result_parts.append("JR=None")
        else:
            print(f"[{scene_name}] Mask GT not found: {gt_mask_dir}")
            scene_result_parts.append("JM=NoGT")
            scene_result_parts.append("JR=NoGT")
            
        # 4.3 ProPainter inpainting.
        try:
            run_propainter(scene_dir, final_scene_mask_dir, inpainting_out_dir)
        except subprocess.CalledProcessError as e:
            print(f"ProPainter failed for {scene_name}: {e}")
            continue
            
        run_summary_parts.append("|".join(scene_result_parts))

    append_result_line(RESULT_TXT_PATH, " || ".join(run_summary_parts))
    print(f"Metrics appended to: {RESULT_TXT_PATH}")
    print(f"\nAll processing complete. Results saved to: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    args = parse_args()
    configure_from_args(args)
    main_pipeline(args.scene)
