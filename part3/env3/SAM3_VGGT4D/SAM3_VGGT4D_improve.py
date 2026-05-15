# Attempt to optimize part 3, the difference is not significant
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

SAM3_MODEL_DIR = str(PROJECT_ROOT / "external/sam3/ckpts")
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
# 2. 评价指标计算函数
# ==========================================
def calculate_iou(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    
    # 修复点 1：完美预测纯静态背景的帧，给予满分 1.0
    if union == 0:
        return 1.0
        
    return intersection / union

def evaluate_mask_quality(pred_masks_dir, gt_mask_dir, threshold=0.5):
    # 修复点 2：放弃 zip，改用字典映射，严格通过“文件名”来对齐帧
    pred_mask_dict = {os.path.basename(p): p for p in glob.glob(os.path.join(pred_masks_dir, "*.png"))}
    gt_mask_dict = {os.path.basename(p): p for p in glob.glob(os.path.join(gt_mask_dir, "*.png"))}
    # 只提取两者都有的帧，避免因首尾帧缺失导致的整体错位
    common_files = sorted(list(set(pred_mask_dict.keys()) & set(gt_mask_dict.keys())))
    if not common_files:
        print("  [警告] 预测掩码与GT掩码没有重合的文件名，请检查目录！")
        return None, None
        
    ious = []
    for file_name in common_files:
        pred_path = pred_mask_dict[file_name]
        gt_path = gt_mask_dict[file_name] 
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  
        if pred_mask is None or gt_mask is None:
            continue      
        # 统一缩放对齐
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(calculate_iou(pred_mask, gt_mask))   
    if not ious:
        return None, None
    # 修复点 3：乘以 100 转为百分制，严格对齐论文中的分数格式
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=Path, default=INPUT_PARENT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_BASE_DIR)

    parser.add_argument(
        "--scene",
        type=str,
        default="both",
        choices=["bmx-trees", "tennis", "both"],
        help="选择运行哪个场景: bmx-trees / tennis / both"
    )

    parser.add_argument("--vggt_ckpt", type=str, default=str(PROJECT_ROOT / "external/VGGT4D/ckpts/model_tracker_fixed_e20.pt"))

    parser.add_argument("--sam3_ckpt", type=str, default=SAM3_CKPT_PATH)
    parser.add_argument("--sam3_bpe", type=str, default=SAM3_BPE_PATH)

    parser.add_argument("--gt_bmx", type=str, default=GT_MASK_DIRS["bmx-trees"])
    parser.add_argument("--gt_tennis", type=str, default=GT_MASK_DIRS["tennis"])

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)

    return parser.parse_args()


def configure_from_args(args):
    global INPUT_PARENT_DIR, OUTPUT_BASE_DIR, RESULT_TXT_PATH
    global SAM3_CKPT_PATH, SAM3_BPE_PATH
    global SCENES_TO_RUN, GT_MASK_DIRS
    global CHUNK_SIZE

    INPUT_PARENT_DIR = Path(args.input_dir)

    OUTPUT_BASE_DIR = Path(args.output_dir)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    RESULT_TXT_PATH = OUTPUT_BASE_DIR / "result.txt"
    RESULT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

    SAM3_CKPT_PATH = args.sam3_ckpt
    SAM3_BPE_PATH = args.sam3_bpe

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
# 3. 核心大招：分块（Chunking）工具函数
# ==========================================
def prepare_chunked_dataset(orig_dir, chunked_parent_dir, scene_name, chunk_size=20):
    """
    将长视频切分为多个短 chunk，防止 VGGT4D 计算长时序 Attention 时物理内存爆炸。
    直接原图复制，不做任何降采样处理。
    """
    img_paths = sorted(glob.glob(os.path.join(orig_dir, "*.[jp][pn]g")))
    if not img_paths: return
    
    for idx, p in enumerate(img_paths):
        # 计算当前图片属于哪个 chunk
        chunk_idx = idx // chunk_size
        # 构造类似 bmx-trees_chunk_00 的目录名
        chunk_dir = chunked_parent_dir / f"{scene_name}_chunk_{chunk_idx:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 shutil.copy 直接复制文件，速度远快于 cv2 读写
        shutil.copy(p, str(chunk_dir / os.path.basename(p)))
        
    print(f"  -> {scene_name} 切分为 {len(img_paths)//chunk_size + 1} 个块 (每块 {chunk_size} 帧，原分辨率)")

# ==========================================
# merge_masks
# ==========================================
def merge_masks(orig_dir, mask_parent_dir, scene_name):
    """
    VGGT4D 跑完后，把各个 chunk 的 mask 拼合回一个完整的文件夹。
    如果完整 merged mask 已存在，则直接复用，不再重复 merge。
    """
    final_scene_mask_dir = mask_parent_dir / scene_name
    final_scene_mask_dir.mkdir(parents=True, exist_ok=True)

    orig_images = sorted(glob.glob(os.path.join(orig_dir, "*.[jp][pn]g")))
    expected_num = len(orig_images)

    # 已有完整 merged mask，直接复用
    existing_num = count_image_files(final_scene_mask_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"  -> 场景 {scene_name}: 已存在完整 merged masks，直接复用 ({existing_num}/{expected_num})")
        return final_scene_mask_dir

    chunk_dirs = sorted(mask_parent_dir.glob(f"{scene_name}_chunk_*"))

    all_mask_files = []
    for c_dir in chunk_dirs:
        chunk_masks = sorted((c_dir / "masks").glob("*.png"))
        all_mask_files.extend(chunk_masks)

    print(f"  -> 场景 {scene_name}: 原图共 {len(orig_images)} 帧, 提取到 {len(all_mask_files)} 张掩码")

    for i, mask_file in enumerate(all_mask_files):
        if i >= len(orig_images):
            break

        orig_name = Path(orig_images[i]).stem
        new_mask_name = f"{orig_name}.png"

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, pure_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(str(final_scene_mask_dir / new_mask_name), pure_mask)

    # 这里仍然保留清理 chunk 输出
    for c_dir in chunk_dirs:
        shutil.rmtree(c_dir)

    return final_scene_mask_dir

# ==========================================
# mask_to_sam3
# ==========================================
def _mask_to_sam3_prompts(mask):
    """
    根据 VGGT4D 粗 mask 自动生成 SAM3 所需的 box + 前景点 prompt。
    这样改动最小，不需要改你原有 VGGT4D/ProPainter 逻辑。
    """
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None, None, None

    ys, xs = np.where(fg > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # 给 box 留一点边界余量，避免框太贴边
    pad = 3
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)

    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    # 用最大连通域的质心作为前景点
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
    print(f"\n[SAM3] 正在精修场景: {image_dir.name}")
    refined_mask_dir.mkdir(parents=True, exist_ok=True)

    expected_num = count_image_files(image_dir)
    existing_num = count_image_files(refined_mask_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"[SAM3] {image_dir.name} 已存在完整 refined masks，直接复用 ({existing_num}/{expected_num})")
        return refined_mask_dir
    
    # 关键检查：你这里的 sam3_ms 只是模型文件目录，不是源码目录
    if not os.path.exists(SAM3_CKPT_PATH):
        print(f"SAM3 checkpoint 不存在: {SAM3_CKPT_PATH}")
        shutil.copytree(coarse_mask_dir, refined_mask_dir, dirs_exist_ok=True)
        return refined_mask_dir

    if not os.path.exists(SAM3_BPE_PATH):
        print(f"SAM3 BPE 不存在: {SAM3_BPE_PATH}")
        shutil.copytree(coarse_mask_dir, refined_mask_dir, dirs_exist_ok=True)
        return refined_mask_dir

    try:
        # 直接使用你 conda 环境里已安装的 sam3 源码包
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        # 用“图像模型”来喂给 InteractiveImagePredictor
        # 不再用 build_tracker(...)
        sam_model = build_sam3_image_model(
            checkpoint_path=SAM3_CKPT_PATH,
            bpe_path=SAM3_BPE_PATH,
            load_from_HF=False,
            eval_mode=True,
            device=SAM3_DEVICE,
        )
        predictor = SAM3InteractiveImagePredictor(sam_model)

    except Exception as e:
        print(f"SAM3 初始化失败，将退回只使用 VGGT4D 粗 mask。错误信息: {e}")
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

            # 官方这个 predictor 的 set_image 是先算 image embeddings
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

                # 保底：如果 SAM3 给出空 mask 或过小 mask，就回退 coarse mask
                if refined_mask.sum() == 0 or refined_mask.sum() < 0.1 * coarse_mask.sum():
                    refined_mask = coarse_mask

            cv2.imwrite(str(out_path), refined_mask)

        except Exception as e:
            print(f"[SAM3] {stem} 精修失败，回退 coarse mask。错误: {e}")
            cv2.imwrite(str(out_path), coarse_mask)

        finally:
            # 不同版本方法名可能略有区别，做兼容处理
            try:
                predictor.reset_predictor()
            except Exception:
                try:
                    predictor.reset_image()
                except Exception:
                    pass

    return refined_mask_dir

# ==========================================
# 4. 模型调用函数
# ==========================================
def run_vggt4d(parent_input_dir, parent_output_dir):
    print(f"\n[VGGT4D] 正在处理分块数据集: {parent_input_dir}")
    parent_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXEC, f"{VGGT4D_DIR}/demo_vggt4dzuri.py", 
        "--input_dir", str(parent_input_dir),
        "--output_dir", str(parent_output_dir)
    ]
    subprocess.run(cmd, check=True, cwd=VGGT4D_DIR)

def run_propainter(dataset_path, mask_dir, output_dir):
    print(f"\n[ProPainter] 正在修复完整视频序列: {dataset_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_num = count_image_files(dataset_path)
    existing_num = count_image_files(output_dir)
    if expected_num > 0 and existing_num == expected_num:
        print(f"[ProPainter] {dataset_path.name} 已存在完整输出，直接复用 ({existing_num}/{expected_num})")
        return

    cmd = [
        PYTHON_EXEC, f"{PROPAINTER_DIR}/inference_propainter.py",
        "--video", str(dataset_path),
        "--mask", str(mask_dir),
        "--output", str(output_dir)
    ]
    subprocess.run(cmd, check=True, cwd=PROPAINTER_DIR)

# ==========================================
# 5. 全局批处理主逻辑
# ==========================================
def main_pipeline():
    print(f"{'='*50}\n开始全局分块批处理 (防 SIGKILL 终极版 - 原分辨率)\n输入父目录: {INPUT_PARENT_DIR}\n{'='*50}")
    append_result_to_txt(RESULT_TXT_PATH, "\n" + "=" * 80)
    append_result_to_txt(RESULT_TXT_PATH, f"开始新一轮评估")   
    scene_dirs = [
        d for d in INPUT_PARENT_DIR.iterdir()
        if d.is_dir() and d.name in SCENES_TO_RUN
    ]
    if not scene_dirs: return

    chunked_parent_dir = OUTPUT_BASE_DIR / "chunked_input"
    mask_parent_dir = OUTPUT_BASE_DIR / "vggt4d_masks"

    # 只对“还没有完整 coarse merged mask”的场景跑 VGGT4D
    scenes_need_vggt4d = []

    print("\n>>> 阶段 1: 检查是否需要准备 chunk / 跑 VGGT4D")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        coarse_merged_dir = mask_parent_dir / scene_name

        expected_num = count_image_files(scene_dir)
        existing_num = count_image_files(coarse_merged_dir)

        if expected_num > 0 and existing_num == expected_num:
            print(f"  -> {scene_name}: 已存在完整 coarse masks，跳过 chunk + VGGT4D ({existing_num}/{expected_num})")
        else:
            print(f"  -> {scene_name}: coarse masks 不完整，准备重新跑 VGGT4D")
            prepare_chunked_dataset(scene_dir, chunked_parent_dir, scene_name, chunk_size=CHUNK_SIZE)
            scenes_need_vggt4d.append(scene_dir)

    # 只有真的有缺失场景时才跑 VGGT4D
    if scenes_need_vggt4d:
        try:
            run_vggt4d(chunked_parent_dir, mask_parent_dir)
        except subprocess.CalledProcessError as e:
            print(f"VGGT4D 运行失败: {e}")
            return
        finally:
            if chunked_parent_dir.exists():
                shutil.rmtree(chunked_parent_dir)
    else:
        print("\n>>> 阶段 2: 所有场景 coarse masks 都已存在，跳过 VGGT4D")

    print("\n>>> 阶段 3: 掩码合并 -> SAM3 -> ProPainter")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        inpainting_out_dir = OUTPUT_BASE_DIR / scene_name / "inpainted_results"
        sam3_refined_mask_dir = OUTPUT_BASE_DIR / "sam3_refined_masks" / scene_name

        coarse_merged_dir = mask_parent_dir / scene_name
        expected_num = count_image_files(scene_dir)
        coarse_existing_num = count_image_files(coarse_merged_dir)

        # 若 coarse merged 已完整存在，直接复用；否则才 merge 本轮 VGGT4D chunk 输出
        if expected_num > 0 and coarse_existing_num == expected_num:
            print(f"  -> {scene_name}: 直接复用已有 coarse merged masks ({coarse_existing_num}/{expected_num})")
            final_scene_mask_dir = coarse_merged_dir
        else:
            final_scene_mask_dir = merge_masks(scene_dir, mask_parent_dir, scene_name)

        # 用 SAM3 精修粗 mask（函数内部会自动判断是否已存在）
        final_scene_mask_dir = run_sam3_refine(
            image_dir=scene_dir,
            coarse_mask_dir=final_scene_mask_dir,
            refined_mask_dir=sam3_refined_mask_dir
        )
           
        # 2. 评估掩码
        j_m, j_r = None, None
        gt_mask_dir = GT_MASK_DIRS.get(scene_name, None)
        if gt_mask_dir and os.path.exists(gt_mask_dir):
            j_m, j_r = evaluate_mask_quality(final_scene_mask_dir, gt_mask_dir)
            if j_m is not None:
                print(f"[{scene_name}] Mask Quality -> J_M: {j_m:.4f}, J_R: {j_r:.4f}")
        else:
            print(f"[{scene_name}] 未找到 GT mask 路径: {gt_mask_dir}")

        # 3. 运行 ProPainter
        try:
            run_propainter(scene_dir, final_scene_mask_dir, inpainting_out_dir)
        except subprocess.CalledProcessError as e:
            print(f"ProPainter 修复 {scene_name} 失败: {e}")
            continue

        # 5. 结果追加写入 result.txt
        result_line = (
            f"[{scene_name}] "
            f"J_M: {j_m:.4f} | J_R: {j_r:.4f} | "
            if (j_m is not None and j_r is not None)
            else f"[{scene_name}] 部分指标计算失败 | "
                 f"J_M={j_m}, J_R={j_r}"
        )
        print(result_line)
        append_result_to_txt(RESULT_TXT_PATH, result_line)
        
    print(f"\n全部处理完成！所有结果保存在: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    args = parse_args()
    configure_from_args(args)
    main_pipeline()