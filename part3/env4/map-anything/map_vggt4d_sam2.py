#!/usr/bin/env python3
"""
VGGT4D + Mapanything + sam2
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from time import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from tqdm import tqdm


VGGT4D_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = VGGT4D_ROOT.parent
MAPANYTHING_ROOT = WORKSPACE_ROOT / "map-anything"
SAM2_ROOT = WORKSPACE_ROOT / "sam2"
for root in (VGGT4D_ROOT, MAPANYTHING_ROOT, SAM2_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

if torch.cuda.is_available():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402
try:
    from mapanything.utils.geometry import depthmap_to_world_frame  # noqa: E402
except ImportError:
    depthmap_to_world_frame = None
try:
    from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local  # noqa: E402
except ImportError:
    initialize_mapanything_local = None
try:
    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt4d.masks.dynamic_mask import (  # noqa: E402
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.models.vggt4d import VGGTFor4D  # noqa: E402
from vggt4d.utils.model_utils import inference as vggt4d_inference  # noqa: E402
from vggt4d.utils.model_utils import organize_qk_dict  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


DEFAULT_LOCAL_MAPANYTHING_CONFIG = {
    "path": str(MAPANYTHING_ROOT / "configs/train.yaml"),
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_path": str(MAPANYTHING_ROOT / "ckpt/model.safetensors"),
    "config_json_path": str(MAPANYTHING_ROOT / "config.json"),
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
    "strict": False,
}

DEFAULT_SAM2_CONFIG = {
    "repo_root": str(SAM2_ROOT),
    "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_image_paths(scene_dir: Path) -> List[Path]:
    return sorted(
        [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name,
    )


def resolve_scene_dirs(input_dir: Path) -> List[Path]:
    direct_images = collect_image_paths(input_dir)
    if direct_images:
        return [input_dir]
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and collect_image_paths(p)])


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(mask.astype(np.float32), (size_hw[1], size_hw[0]), interpolation=interp)


def robust_norm(x: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if valid is not None:
        finite &= valid.astype(bool)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[finite], [2.0, 98.0])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def load_vggt4d_model(ckpt_path: Path, device: torch.device) -> VGGTFor4D:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"VGGT4D checkpoint not found: {ckpt_path}. "
            "Download it or pass --vggt_ckpt."
        )
    model = VGGTFor4D()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def parse_mapanything_local_config(raw: Optional[str]) -> dict:
    cfg = dict(DEFAULT_LOCAL_MAPANYTHING_CONFIG)
    if raw:
        user_cfg = json.loads(raw)
        cfg.update(user_cfg)
    config_json_path = cfg.get("config_json_path")
    if config_json_path and not Path(config_json_path).exists():
        cfg.pop("config_json_path", None)
    return cfg


def load_mapanything_model(args: argparse.Namespace, device: torch.device):
    if args.mapanything_local_config is not None:
        if initialize_mapanything_local is None:
            raise ImportError(
                "This map-anything checkout does not provide "
                "mapanything.utils.hf_utils.hf_helpers.initialize_mapanything_local. "
                "Use HuggingFace loading, or update the local-weight loader for this checkout."
            )
        config = parse_mapanything_local_config(args.mapanything_local_config)
        return initialize_mapanything_local(config, device)

    model_name = args.mapanything_model_name
    if args.apache:
        model_name = "facebook/map-anything-apache"
    return MapAnything.from_pretrained(model_name).to(device).eval()


def resolve_optional_path(path_value: Optional[str], root_value: Optional[str]) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if root_value is not None:
        root = Path(root_value)
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def resolve_sam2_config_name(cfg_path: Path, repo_root: Path) -> str:
    cfg_path = cfg_path.resolve()
    repo_root = repo_root.resolve()
    for base in (repo_root / "sam2", repo_root):
        try:
            rel = cfg_path.relative_to(base.resolve())
            return rel.as_posix()
        except ValueError:
            continue
    return cfg_path.as_posix()


def load_sam2_predictor(args: argparse.Namespace, device: torch.device):
    if not args.sam2_refine:
        return None
    if SAM2ImagePredictor is None or build_sam2 is None:
        raise ImportError(
            "SAM2 is not available. Install the official sam2 package or clone the repo "
            "into ../sam2 and install it in the current environment."
        )

    repo_root = Path(args.sam2_repo_root or DEFAULT_SAM2_CONFIG["repo_root"]).resolve()
    cfg_path = resolve_optional_path(args.sam2_cfg, str(repo_root))
    ckpt_path = resolve_optional_path(args.sam2_ckpt, str(repo_root))
    if cfg_path is None or ckpt_path is None:
        raise ValueError("SAM2 refinement requires both --sam2_cfg and --sam2_ckpt.")
    if not cfg_path.exists():
        raise FileNotFoundError(f"SAM2 config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    cfg_name = resolve_sam2_config_name(cfg_path, repo_root)
    sam2_model = build_sam2(cfg_name, str(ckpt_path), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def call_mapanything_infer(model, views: List[dict], args: argparse.Namespace) -> List[dict]:
    infer_kwargs = {
        "memory_efficient_inference": True,
        "minibatch_size": args.minibatch_size,
        "use_amp": not args.no_amp,
        "amp_dtype": args.amp_dtype,
        "apply_mask": True,
        "mask_edges": True,
        "edge_normal_threshold": args.edge_normal_threshold,
        "edge_depth_threshold": args.edge_depth_threshold,
        "apply_confidence_mask": args.apply_confidence_mask,
        "confidence_percentile": args.confidence_percentile,
        "use_multiview_confidence": args.use_multiview_confidence,
        "multiview_conf_depth_abs_thresh": args.mv_depth_abs_thresh,
        "multiview_conf_depth_rel_thresh": args.mv_depth_rel_thresh,
    }
    try:
        valid_params = set(inspect.signature(model.infer).parameters)
        infer_kwargs = {k: v for k, v in infer_kwargs.items() if k in valid_params}
    except (TypeError, ValueError):
        pass
    return model.infer(views, **infer_kwargs)


@torch.no_grad()
def compute_vggt4d_dynamic_prior(
    model: VGGTFor4D,
    image_paths: Sequence[Path],
    device: torch.device,
    preprocess_mode: str,
    n_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode).to(device)
    predictions, qk_dict, enc_feat, agg_tokens = vggt4d_inference(model, images)
    del predictions, agg_tokens
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])
    dyn_maps = extract_dyn_map(qk_dict, images)

    n_img, _, h_img, w_img = images.shape
    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps, n_clusters=n_clusters)
    upsampled = F.interpolate(
        rearrange(norm_dyn_map, "n h w -> n 1 h w"),
        size=(h_img, w_img),
        mode="bilinear",
        align_corners=False,
    )
    upsampled = rearrange(upsampled, "n 1 h w -> n h w").clamp(0, 1).cpu()
    threshold = adaptive_multiotsu_variance(upsampled.numpy())
    coarse_masks = upsampled > threshold
    return images.cpu(), upsampled, coarse_masks


def infer_mapanything(args: argparse.Namespace, model, image_paths: Sequence[Path]) -> List[dict]:
    views = load_images(
        [str(p) for p in image_paths],
        resize_mode=args.map_resize_mode,
        size=args.map_size,
        norm_type=args.map_norm_type,
        patch_size=args.map_patch_size,
        resolution_set=args.map_resolution,
        stride=args.stride,
        verbose=args.verbose,
    )
    if len(views) == 0:
        raise ValueError("No valid images found for MapAnything inference")
    return call_mapanything_infer(model, views, args)


def mapanything_arrays(outputs: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images, depths, intrinsics, cam2world, valid_masks, confs = [], [], [], [], [], []
    for pred in outputs:
        depth_t = pred["depth_z"][0].squeeze(-1)
        intr_t = pred["intrinsics"][0]
        pose_t = pred["camera_poses"][0]
        if depthmap_to_world_frame is not None:
            _, valid_depth_t = depthmap_to_world_frame(depth_t, intr_t, pose_t)
            valid_depth = valid_depth_t.cpu().numpy().astype(bool)
        else:
            valid_depth = torch.isfinite(depth_t).cpu().numpy().astype(bool)
            valid_depth &= depth_t.cpu().numpy() > 0

        mask_t = pred.get("mask")
        if mask_t is None:
            valid_mask = valid_depth
        else:
            valid_mask = mask_t[0].squeeze(-1).cpu().numpy().astype(bool)
            valid_mask &= valid_depth

        conf_t = pred.get("conf")
        if conf_t is None:
            conf = np.ones(depth_t.shape, dtype=np.float32)
        else:
            conf = conf_t[0].detach().cpu().numpy().astype(np.float32)

        images.append(pred["img_no_norm"][0].detach().cpu().numpy().astype(np.float32))
        depths.append(depth_t.detach().cpu().numpy().astype(np.float32))
        intrinsics.append(intr_t.detach().cpu().numpy().astype(np.float32))
        cam2world.append(pose_t.detach().cpu().numpy().astype(np.float32))
        valid_masks.append(valid_mask)
        confs.append(conf)

    return (
        np.stack(images, axis=0),
        np.stack(depths, axis=0),
        np.stack(intrinsics, axis=0),
        np.stack(cam2world, axis=0),
        np.stack(valid_masks, axis=0),
        np.stack(confs, axis=0),
    )


def inverse_project_np(depths: np.ndarray, intrinsics: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    n, h, w = depths.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5, np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
    pix = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    world = np.empty((n, h, w, 3), dtype=np.float32)
    for i in range(n):
        cam = pix * depths[i, ..., None]
        cam = cam @ np.linalg.inv(intrinsics[i]).T
        world[i] = cam @ cam2world[i, :3, :3].T + cam2world[i, :3, 3]
    return world


def project_points_np(points: np.ndarray, intrinsics: np.ndarray, world2cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cam = points @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = cam[:, 2]
    uv = cam @ intrinsics.T
    uv[:, 0] /= uv[:, 2] + 1e-8
    uv[:, 1] /= uv[:, 2] + 1e-8
    return uv[:, :2], z


def sample_image_np(arr: np.ndarray, uv: np.ndarray, nearest: bool = False) -> np.ndarray:
    h, w = arr.shape[:2]
    x = uv[:, 0].astype(np.float32)
    y = uv[:, 1].astype(np.float32)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    sampled = cv2.remap(
        arr,
        x.reshape(1, -1),
        y.reshape(1, -1),
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape((-1,) + arr.shape[2:])


def largest_components(mask: np.ndarray, keep: int, min_area: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    out = np.zeros_like(mask, dtype=bool)
    for idx in order[:keep]:
        if areas[idx] >= min_area:
            out |= labels == (idx + 1)
    return out


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return mask
    inv = (~mask.astype(bool)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.astype(bool).copy()
    h, w = mask.shape
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h
        if not touches_border:
            out[labels == label] = True
    return out


def component_filter_by_score(mask: np.ndarray, score: np.ndarray, min_area: int, min_score: float) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, n_labels):
        region = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if float(score[region].mean()) < min_score:
            continue
        out |= region
    return out


def otsu_score_threshold(score: np.ndarray, floor: float, scale: float = 0.75) -> float:
    score_u8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)
    thres, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(floor, float(thres / 255.0) * scale)


def compute_image_gradient(image: np.ndarray) -> np.ndarray:
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.GaussianBlur(grad, (3, 3), 0)
    return robust_norm(grad)


def boundary_alignment_score(mask: np.ndarray, image_grad: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ero = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = (dil > ero)
    if boundary.sum() == 0:
        return 0.0
    return float(image_grad[boundary].mean())


def contour_depth(hierarchy: np.ndarray, idx: int) -> int:
    depth = 0
    parent = int(hierarchy[idx][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def smooth_final_mask(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    if args.final_smooth_close_kernel > 0:
        k = max(3, int(args.final_smooth_close_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=args.final_smooth_close_iters)
    if args.final_smooth_open_kernel > 0:
        k = max(3, int(args.final_smooth_open_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=args.final_smooth_open_iters)
    if args.final_smooth_blur_kernel > 0:
        k = max(3, int(args.final_smooth_blur_kernel) | 1)
        mask_blur = cv2.GaussianBlur(mask_u8.astype(np.float32), (k, k), 0)
        _, mask_u8 = cv2.threshold(mask_blur, args.final_smooth_threshold, 255, cv2.THRESH_BINARY)
        mask_u8 = mask_u8.astype(np.uint8)
    mask_bool = fill_small_holes(mask_u8 > 0, args.final_smooth_fill_hole_area)

    contours_info = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_info) == 3:
        _, contours, hierarchy = contours_info
    else:
        contours, hierarchy = contours_info

    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]
        poly_mask = np.zeros_like(mask_u8)
        for idx, contour in enumerate(contours):
            area = abs(float(cv2.contourArea(contour)))
            if area < args.final_poly_min_area:
                continue
            epsilon = max(1.0, float(cv2.arcLength(contour, True)) * args.final_poly_epsilon_ratio)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            depth = contour_depth(hierarchy, idx)
            color = 255 if (depth % 2 == 0) else 0
            cv2.drawContours(poly_mask, [approx], -1, color=color, thickness=-1)
        if poly_mask.any():
            mask_u8 = poly_mask

    mask_bool = mask_u8 > 0
    mask_bool = fill_small_holes(mask_bool, args.final_smooth_fill_hole_area)
    if args.keep_largest_components > 0:
        mask_bool = largest_components(mask_bool, args.keep_largest_components, args.prior_min_area)
    return mask_bool


def mask_to_box(mask: np.ndarray, pad: int, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = image_shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w - 1, int(xs.max()) + pad)
    y1 = min(h - 1, int(ys.max()) + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def sample_prompt_points(mask: np.ndarray, n_pos: int, n_neg: int) -> Tuple[np.ndarray, np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    h, w = mask.shape
    if mask_u8.sum() == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    dist_fg = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    pos_coords = []
    pos_labels = []
    work = dist_fg.copy()
    for _ in range(max(1, n_pos)):
        idx = np.unravel_index(np.argmax(work), work.shape)
        if work[idx] <= 0:
            break
        y, x = idx
        pos_coords.append([float(x), float(y)])
        pos_labels.append(1)
        cv2.circle(work, (int(x), int(y)), 12, 0, -1)

    ring = cv2.dilate(mask_u8, np.ones((17, 17), np.uint8), iterations=1) > 0
    ring &= ~mask.astype(bool)
    neg_coords = []
    neg_labels = []
    if ring.any():
        dist_neg = cv2.distanceTransform(ring.astype(np.uint8), cv2.DIST_L2, 5)
        work = dist_neg.copy()
        for _ in range(max(1, n_neg)):
            idx = np.unravel_index(np.argmax(work), work.shape)
            if work[idx] <= 0:
                break
            y, x = idx
            neg_coords.append([float(x), float(y)])
            neg_labels.append(0)
            cv2.circle(work, (int(x), int(y)), 12, 0, -1)
    if not neg_coords:
        neg_coords = [[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]]
        neg_labels = [0, 0, 0, 0]

    coords = np.array(pos_coords + neg_coords, dtype=np.float32)
    labels = np.array(pos_labels + neg_labels, dtype=np.int32)
    return coords, labels


def sam2_refine_one(
    predictor,
    image: np.ndarray,
    baseline_mask: np.ndarray,
    prior_score: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    baseline_mask = baseline_mask.astype(bool)
    if baseline_mask.sum() < args.sam2_min_area:
        return baseline_mask

    box = mask_to_box(baseline_mask, args.sam2_box_pad, baseline_mask.shape)
    if box is None:
        return baseline_mask
    point_coords, point_labels = sample_prompt_points(
        baseline_mask,
        n_pos=args.sam2_num_positive_points,
        n_neg=args.sam2_num_negative_points,
    )
    if point_coords.shape[0] == 0:
        return baseline_mask

    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    predictor.set_image(img_u8)
    try:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None, :],
            multimask_output=True,
        )
    except TypeError:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    candidates = np.asarray(masks).astype(bool)
    scores = np.asarray(scores).reshape(-1)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]

    grad = compute_image_gradient(image)
    best_mask = baseline_mask
    best_value = -1e9
    prior_area = float(baseline_mask.sum())
    for cand, sam_score in zip(candidates, scores):
        cand = component_filter_by_score(
            cand,
            prior_score,
            min_area=args.sam2_min_area,
            min_score=args.prior_component_min_score,
        )
        if cand.sum() == 0:
            continue
        inter = float((cand & baseline_mask).sum())
        union = float((cand | baseline_mask).sum())
        iou = inter / (union + 1e-6)
        area_ratio = float(cand.sum()) / (prior_area + 1e-6)
        edge_gain = boundary_alignment_score(cand, grad) - boundary_alignment_score(baseline_mask, grad)
        if iou < args.sam2_min_iou_with_prior:
            continue
        if area_ratio < args.sam2_min_area_ratio or area_ratio > args.sam2_max_area_ratio:
            continue
        value = float(sam_score) + args.sam2_edge_gain_weight * edge_gain + args.sam2_iou_weight * iou
        if value > best_value:
            best_value = value
            best_mask = cand
    return best_mask


def sam2_refine_masks(
    predictor,
    images: np.ndarray,
    baseline_masks: np.ndarray,
    prior_scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if predictor is None:
        return baseline_masks.astype(bool)
    refined = []
    for i in tqdm(range(len(baseline_masks)), desc="SAM2 refine"):
        refined.append(
            sam2_refine_one(
                predictor,
                images[i],
                baseline_masks[i],
                prior_scores[i],
                args,
            )
        )
    return np.stack(refined, axis=0)


def edge_aware_refine_one(
    image: np.ndarray,
    baseline_mask: np.ndarray,
    prior_score: np.ndarray,
    geometry_score: np.ndarray,
    geometry_mask: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    baseline_mask = baseline_mask.astype(bool)
    valid_mask = valid_mask.astype(bool)
    combined_score = np.maximum(prior_score, geometry_score).astype(np.float32)
    support_thres = otsu_score_threshold(
        prior_score,
        floor=args.edge_refine_prior_floor,
        scale=args.edge_refine_otsu_scale,
    )
    support = baseline_mask | (prior_score >= support_thres)
    support &= valid_mask
    support = component_filter_by_score(
        support,
        prior_score,
        min_area=args.edge_refine_min_area,
        min_score=max(args.edge_refine_component_min_score, 0.5 * args.prior_component_min_score),
    )
    if support.sum() < args.edge_refine_min_area:
        return baseline_mask

    band_kernel = max(3, int(args.grabcut_band_kernel) | 1)
    sure_kernel = max(3, int(args.grabcut_sure_fg_kernel) | 1)
    band = cv2.dilate(
        support.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_kernel, band_kernel)),
        iterations=args.grabcut_band_iters,
    ).astype(bool)
    band &= valid_mask
    sure_fg = cv2.erode(
        baseline_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sure_kernel, sure_kernel)),
        iterations=1,
    ).astype(bool)
    sure_fg |= (prior_score >= args.edge_refine_sure_fg_score) & support
    sure_fg |= geometry_mask.astype(bool) & baseline_mask
    sure_fg &= support & valid_mask

    if sure_fg.sum() < args.edge_refine_min_area:
        sure_fg = baseline_mask.copy()

    gc_mask = np.full(support.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[band] = cv2.GC_PR_BGD
    gc_mask[support] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD
    far_bg = band & (~support) & (geometry_score < args.edge_refine_bg_score)
    gc_mask[far_bg] = cv2.GC_BGD

    if len(np.unique(gc_mask)) < 2 or not np.any(gc_mask == cv2.GC_FGD):
        refined = baseline_mask
    else:
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_u8, gc_mask, None, bgd_model, fgd_model, args.grabcut_iters, cv2.GC_INIT_WITH_MASK)
            refined = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            refined &= band
        except cv2.error:
            refined = baseline_mask

    if args.edge_refine_close_kernel > 0:
        k = max(3, int(args.edge_refine_close_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    refined = fill_small_holes(refined, args.edge_refine_fill_hole_area)
    refined = component_filter_by_score(
        refined,
        prior_score,
        min_area=args.edge_refine_min_area,
        min_score=args.edge_refine_component_min_score,
    )
    if args.keep_largest_components > 0:
        refined = largest_components(refined, args.keep_largest_components, args.edge_refine_min_area)
    return refined.astype(bool)


def edge_aware_refine_masks(
    images: np.ndarray,
    baseline_masks: np.ndarray,
    prior_scores: np.ndarray,
    geometry_scores: np.ndarray,
    geometry_masks: np.ndarray,
    valid_masks: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if not args.edge_refine:
        return geometry_masks.astype(bool)
    masks = []
    for i in tqdm(range(len(geometry_masks)), desc="RGB edge refine"):
        masks.append(
            edge_aware_refine_one(
                images[i],
                baseline_masks[i],
                prior_scores[i],
                geometry_scores[i],
                geometry_masks[i],
                valid_masks[i],
                args,
            )
        )
    return np.stack(masks, axis=0)


def conservative_prior_masks(
    prior_scores: np.ndarray,
    prior_masks: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    masks = []
    for score, mask in zip(prior_scores, prior_masks):
        support = mask.astype(bool) | (score >= args.prior_rescue_score)
        support = component_filter_by_score(
            support,
            score,
            min_area=args.prior_min_area,
            min_score=args.prior_component_min_score,
        )
        if args.prior_close_kernel > 0:
            k = max(3, int(args.prior_close_kernel) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
        support = fill_small_holes(support, args.prior_fill_hole_area)
        if args.keep_largest_components > 0:
            support = largest_components(support, args.keep_largest_components, args.prior_min_area)
        masks.append(support.astype(bool))
    return np.stack(masks, axis=0)


def guarded_select_masks(
    images: np.ndarray,
    baseline_masks: np.ndarray,
    candidate_masks: np.ndarray,
    fallback_masks: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    selected = []
    for image, prior, candidate, fallback in zip(images, baseline_masks, candidate_masks, fallback_masks):
        prior = prior.astype(bool)
        candidate = candidate.astype(bool)
        fallback = fallback.astype(bool)
        prior_area = float(prior.sum())
        candidate_area = float(candidate.sum())
        if prior_area < 1:
            selected.append(candidate)
            continue
        inter = float((prior & candidate).sum())
        union = float((prior | candidate).sum())
        iou = inter / (union + 1e-6)
        area_ratio = candidate_area / (prior_area + 1e-6)
        grad = compute_image_gradient(image)
        prior_edge = boundary_alignment_score(prior, grad)
        cand_edge = boundary_alignment_score(candidate, grad)
        edge_gain = cand_edge - prior_edge
        if iou < args.guard_min_iou_with_prior:
            selected.append(fallback)
        elif area_ratio < args.guard_min_area_ratio or area_ratio > args.guard_max_area_ratio:
            selected.append(fallback)
        elif edge_gain < args.guard_min_boundary_gain:
            selected.append(fallback)
        else:
            selected.append(candidate)
    return np.stack(selected, axis=0)


def score_mask_candidate(
    image: np.ndarray,
    prior_score: np.ndarray,
    baseline_mask: np.ndarray,
    candidate_mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    geometry_mask: Optional[np.ndarray],
    args: argparse.Namespace,
) -> float:
    candidate_mask = candidate_mask.astype(bool)
    baseline_mask = baseline_mask.astype(bool)
    if valid_mask is not None:
        candidate_mask &= valid_mask.astype(bool)
    if candidate_mask.sum() == 0:
        return -1e9

    prior_area = float(baseline_mask.sum())
    cand_area = float(candidate_mask.sum())
    inter = float((candidate_mask & baseline_mask).sum())
    union = float((candidate_mask | baseline_mask).sum())
    iou = inter / (union + 1e-6)
    area_ratio = cand_area / (prior_area + 1e-6)
    mean_prior = float(prior_score[candidate_mask].mean()) if cand_area > 0 else 0.0

    grad = compute_image_gradient(image)
    baseline_edge = boundary_alignment_score(baseline_mask, grad)
    candidate_edge = boundary_alignment_score(candidate_mask, grad)
    edge_gain = candidate_edge - baseline_edge

    geometry_iou = 0.0
    if geometry_mask is not None and geometry_mask.sum() > 0:
        geometry_mask = geometry_mask.astype(bool)
        geo_inter = float((candidate_mask & geometry_mask).sum())
        geo_union = float((candidate_mask | geometry_mask).sum())
        geometry_iou = geo_inter / (geo_union + 1e-6)

    area_penalty = abs(np.log(np.clip(area_ratio, 1e-4, 1e4)))
    score = (
        0.42 * mean_prior
        + 0.24 * iou
        + 0.16 * edge_gain
        + 0.10 * geometry_iou
        - 0.10 * area_penalty
    )

    if mean_prior < 0.10:
        score -= 0.25
    if iou < max(0.35, args.guard_min_iou_with_prior - 0.35):
        score -= 0.35
    if area_ratio < args.guard_min_area_ratio * 0.70 or area_ratio > args.guard_max_area_ratio * 1.30:
        score -= 0.30
    return float(score)


def select_best_masks(
    images: np.ndarray,
    prior_scores: np.ndarray,
    baseline_masks: np.ndarray,
    candidate_masks: List[Tuple[str, np.ndarray]],
    valid_masks: Optional[np.ndarray],
    geometry_masks: Optional[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    selected = []
    choice_counts = {name: 0 for name, _ in candidate_masks}
    for i in range(len(baseline_masks)):
        best_mask = None
        best_name = candidate_masks[0][0]
        best_score = -1e9
        geometry_mask = None if geometry_masks is None else geometry_masks[i]
        valid_mask = None if valid_masks is None else valid_masks[i]
        for name, masks in candidate_masks:
            candidate_mask = smooth_final_mask(masks[i], args)
            score = score_mask_candidate(
                images[i],
                prior_scores[i],
                baseline_masks[i],
                candidate_mask,
                valid_mask,
                geometry_mask,
                args,
            )
            if score > best_score:
                best_score = score
                best_name = name
                best_mask = candidate_mask
        assert best_mask is not None
        selected.append(best_mask)
        choice_counts[best_name] += 1
    print("  Candidate selection:", ", ".join(f"{name}={count}" for name, count in choice_counts.items()))
    return np.stack(selected, axis=0)


class MapGeometryMaskRefiner:
    def __init__(
        self,
        images: np.ndarray,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        cam2world: np.ndarray,
        valid_masks: np.ndarray,
        confs: np.ndarray,
        args: argparse.Namespace,
    ) -> None:
        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.cam2world = cam2world
        self.world2cam = np.linalg.inv(cam2world)
        self.valid_masks = valid_masks.astype(bool)
        self.confs = np.stack([robust_norm(c, v) for c, v in zip(confs, self.valid_masks)], axis=0)
        self.args = args
        self.points = inverse_project_np(depths, intrinsics, cam2world)

    def refine(self, coarse_scores: np.ndarray, coarse_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        refined = []
        refined_scores = []
        for cam_id in tqdm(range(len(coarse_masks)), desc="MapAnything geometry refine"):
            score, mask = self.refine_one(cam_id, coarse_scores[cam_id], coarse_masks[cam_id])
            refined_scores.append(score)
            refined.append(mask)
        return np.stack(refined_scores, axis=0), np.stack(refined, axis=0)

    def refine_one(self, cam_id: int, coarse_score: np.ndarray, coarse_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid = self.valid_masks[cam_id]
        conf = self.confs[cam_id]
        candidate = coarse_mask.astype(bool)
        candidate &= (conf >= self.args.min_map_conf) | (coarse_score >= self.args.prior_rescue_score)

        score = coarse_score.astype(np.float32).copy()
        if candidate.sum() < self.args.min_region_area:
            final = self.postprocess(candidate)
            return score, final

        labels, n_labels = self.segment_candidates(cam_id, candidate, coarse_score)
        keep = np.zeros(n_labels + 1, dtype=bool)
        for label in range(1, n_labels + 1):
            region = labels == label
            if region.sum() < self.args.min_region_area:
                continue
            dyn_score, support = self.region_dynamic_score(cam_id, region)
            region_score = (
                self.args.geometry_weight * dyn_score
                + self.args.prior_weight * float(coarse_score[region].mean())
                + self.args.conf_weight * float(conf[region].mean())
            )
            if support == 0:
                region_score = max(region_score, float(coarse_score[region].mean()))
            elif support < self.args.min_projection_support:
                region_score *= 0.75
            if region_score >= self.args.keep_score:
                keep[label] = True
            score[region] = np.clip(region_score, 0.0, 1.0)

        final = keep[labels]
        final &= candidate
        final = self.postprocess(final)
        return score, final

    def segment_candidates(self, cam_id: int, candidate: np.ndarray, coarse_score: np.ndarray) -> Tuple[np.ndarray, int]:
        n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        labels = np.zeros_like(cc_labels, dtype=np.int32)
        next_label = 1
        for cc in range(1, n_cc):
            cc_mask = cc_labels == cc
            area = int(stats[cc, cv2.CC_STAT_AREA])
            if area < self.args.min_region_area:
                continue
            if area < self.args.cluster_split_area:
                labels[cc_mask] = next_label
                next_label += 1
                continue

            ys, xs = np.nonzero(cc_mask)
            pts = self.points[cam_id, ys, xs]
            rgb = self.images[cam_id, ys, xs]
            dyn = coarse_score[ys, xs, None]
            feat = np.concatenate(
                [
                    pts / (np.nanstd(pts, axis=0, keepdims=True) + 1e-6),
                    rgb,
                    dyn,
                ],
                axis=1,
            )
            n_clusters = min(self.args.max_region_clusters, max(2, area // self.args.cluster_split_area + 1))
            try:
                sub = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(feat)
            except TypeError:
                sub = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(feat)
            for sub_id in range(n_clusters):
                sub_mask = sub == sub_id
                if int(sub_mask.sum()) >= self.args.min_region_area:
                    labels[ys[sub_mask], xs[sub_mask]] = next_label
                    next_label += 1
        return labels, next_label - 1

    def region_dynamic_score(self, cam_id: int, region: np.ndarray) -> Tuple[float, int]:
        ys, xs = np.nonzero(region)
        if len(xs) > self.args.max_points_per_region:
            rng = np.random.default_rng(42 + cam_id)
            keep = rng.choice(len(xs), size=self.args.max_points_per_region, replace=False)
            ys, xs = ys[keep], xs[keep]

        points = self.points[cam_id, ys, xs]
        rgb = self.images[cam_id, ys, xs]
        h, w = self.depths.shape[1:]
        static_votes = []
        dynamic_votes = []

        for other_id in range(len(self.depths)):
            if other_id == cam_id:
                continue
            uv, z = project_points_np(points, self.intrinsics[other_id], self.world2cam[other_id])
            in_frame = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1) & (z > 0)
            if in_frame.sum() < self.args.min_points_per_projection:
                continue

            uv_v = uv[in_frame]
            z_v = z[in_frame]
            rgb_v = rgb[in_frame]
            depth_v = sample_image_np(self.depths[other_id], uv_v)
            valid_v = sample_image_np(self.valid_masks[other_id].astype(np.float32), uv_v, nearest=True) > 0.5
            conf_v = sample_image_np(self.confs[other_id], uv_v)
            img_v = sample_image_np(self.images[other_id], uv_v)

            visible = z_v <= depth_v + np.maximum(self.args.depth_abs_tol, self.args.depth_rel_tol * np.maximum(depth_v, 1e-6))
            reliable = valid_v & (conf_v >= self.args.min_map_conf) & visible
            if reliable.sum() < self.args.min_points_per_projection:
                continue

            depth_err = np.abs(z_v - depth_v) / np.maximum(depth_v, 1e-6)
            rgb_err = np.mean(np.abs(rgb_v - img_v), axis=-1)
            consistent = (depth_err <= self.args.static_depth_rel_tol) & (rgb_err <= self.args.static_rgb_tol)
            static_votes.append(float((consistent & reliable).sum()) / float(reliable.sum()))
            dynamic_votes.append(1.0 - static_votes[-1])

        if not dynamic_votes:
            return 0.0, 0

        # A truly dynamic region should fail static cross-view consistency in
        # most usable projections, but the median resists occasional bad poses.
        return float(np.median(dynamic_votes)), len(dynamic_votes)

    def postprocess(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8) * 255
        if self.args.close_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.close_kernel, self.args.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.args.close_iters)
        if self.args.open_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.open_kernel, self.args.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        if self.args.dilate_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.dilate_kernel, self.args.dilate_kernel))
            mask = cv2.dilate(mask, k, iterations=self.args.dilate_iters)
        out = mask > 0
        if self.args.keep_largest_components > 0:
            out = largest_components(out, self.args.keep_largest_components, self.args.min_region_area)
        return out


def save_tum_poses(path: Path, cam2world: np.ndarray) -> None:
    with path.open("w") as f:
        for i, c2w in enumerate(cam2world):
            quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            qx, qy, qz, qw = quat_xyzw
            xyz = c2w[:3, 3]
            f.write(f"{float(i)} {xyz[0]} {xyz[1]} {xyz[2]} {qw} {qx} {qy} {qz}\n")


def save_outputs(
    output_dir: Path,
    refined_masks: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(refined_masks)):
        refined = smooth_final_mask(refined_masks[i], args)
        cv2.imwrite(str(output_dir / f"dynamic_mask_{i:04d}.png"), refined.astype(np.uint8) * 255)


def align_priors_to_mapanything(
    prior_scores: torch.Tensor,
    prior_masks: torch.Tensor,
    map_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    scores = []
    masks = []
    for score, mask in zip(prior_scores.numpy(), prior_masks.numpy()):
        score_r = resize_mask(score, map_shape_hw, nearest=False)
        mask_r = resize_mask(mask.astype(np.float32), map_shape_hw, nearest=True) > 0.5
        scores.append(robust_norm(score_r))
        masks.append(mask_r)
    return np.stack(scores, axis=0), np.stack(masks, axis=0)


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    vggt4d_model: VGGTFor4D,
    mapanything_model,
    sam2_predictor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    image_paths = collect_image_paths(scene_dir)
    if args.stride > 1:
        image_paths = image_paths[:: args.stride]
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if len(image_paths) == 0:
        print(f"Skipping {scene_dir}: no images")
        return

    print(f"\nProcessing {scene_dir.name}: {len(image_paths)} frames")
    scene_start = time()
    print("  Stage 1/3: VGGT4D dynamic prior")
    _, prior_scores_t, prior_masks_t = compute_vggt4d_dynamic_prior(
        vggt4d_model,
        image_paths,
        device,
        args.vggt4d_preprocess_mode,
        args.vggt4d_clusters,
    )
    torch.cuda.empty_cache()

    print("  Stage 2/3: MapAnything geometry")
    map_outputs = infer_mapanything(args, mapanything_model, image_paths)
    images, depths, intrinsics, cam2world, valid_masks, confs = mapanything_arrays(map_outputs)

    if len(images) != len(prior_scores_t):
        n = min(len(images), len(prior_scores_t))
        print(f"  Warning: frame count mismatch; using first {n} frames")
        images, depths, intrinsics, cam2world, valid_masks, confs = (
            images[:n],
            depths[:n],
            intrinsics[:n],
            cam2world[:n],
            valid_masks[:n],
            confs[:n],
        )
        prior_scores_t = prior_scores_t[:n]
        prior_masks_t = prior_masks_t[:n]

    map_h, map_w = depths.shape[1:]
    prior_scores, prior_masks = align_priors_to_mapanything(prior_scores_t, prior_masks_t, (map_h, map_w))

    print("  Stage 3/3: dynamic-mask refinement")
    baseline_masks = conservative_prior_masks(prior_scores, prior_masks, args)
    sam2_masks = None
    if sam2_predictor is not None:
        sam2_masks = sam2_refine_masks(sam2_predictor, images, baseline_masks, prior_scores, args)
    if len(prior_masks) < args.min_views_for_geometry:
        print("  Not enough views for geometry consistency; using conservative VGGT4D prior.")
        refined_scores = prior_scores
        candidate_masks = [("baseline", baseline_masks)]
        if sam2_masks is not None:
            candidate_masks.append(("sam2", sam2_masks))
        refined_masks = select_best_masks(
            images,
            prior_scores,
            baseline_masks,
            candidate_masks,
            valid_masks=None,
            geometry_masks=None,
            args=args,
        )
    else:
        refiner = MapGeometryMaskRefiner(images, depths, intrinsics, cam2world, valid_masks, confs, args)
        refined_scores, geometry_masks = refiner.refine(prior_scores, prior_masks)
        edge_masks = edge_aware_refine_masks(
            images,
            baseline_masks,
            prior_scores,
            refined_scores,
            geometry_masks,
            valid_masks,
            args,
        )
        candidate_masks = [("baseline", baseline_masks)]
        if sam2_masks is not None:
            candidate_masks.append(("sam2", sam2_masks))
        candidate_masks.append(("geometry_edge", edge_masks))
        refined_masks = select_best_masks(
            images,
            prior_scores,
            baseline_masks,
            candidate_masks,
            valid_masks=valid_masks,
            geometry_masks=geometry_masks,
            args=args,
        )

    print(f"  Saving to {output_dir}")
    save_outputs(
        output_dir,
        refined_masks,
        args,
    )
    print(f"  Done in {time() - scene_start:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse VGGT4D dynamic priors with MapAnything geometry for stronger masks.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Scene folder or folder containing scene subfolders.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root.")
    parser.add_argument("--vggt_ckpt", type=Path, default=Path("./ckpts/model_tracker_fixed_e20.pt"))
    parser.add_argument("--mapanything_model_name", type=str, default="facebook/map-anything")
    parser.add_argument("--mapanything_local_config", type=str, default=None, help="JSON overrides for local MapAnything weights. Passing this enables local loading.")
    parser.add_argument("--apache", action="store_true", help="Use facebook/map-anything-apache when loading from HuggingFace.")
    parser.add_argument("--hf_endpoint", type=str, default=None, help="Optional HuggingFace endpoint mirror.")
    parser.add_argument("--sam2_refine", dest="sam2_refine", action="store_true", default=False)
    parser.add_argument("--sam2_repo_root", type=str, default=DEFAULT_SAM2_CONFIG["repo_root"])
    parser.add_argument("--sam2_cfg", type=str, default=DEFAULT_SAM2_CONFIG["cfg"])
    parser.add_argument("--sam2_ckpt", type=str, default=DEFAULT_SAM2_CONFIG["checkpoint"])

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=20, help="Only process the first N frames. Set <=0 to use all frames.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vggt4d_preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--vggt4d_clusters", type=int, default=64)
    parser.add_argument("--map_resize_mode", choices=["fixed_mapping", "longest_side", "square", "fixed_size"], default="fixed_mapping")
    parser.add_argument("--map_size", type=int, default=None)
    parser.add_argument("--map_norm_type", type=str, default="dinov2")
    parser.add_argument("--map_patch_size", type=int, default=14)
    parser.add_argument("--map_resolution", type=int, default=518)
    parser.add_argument("--minibatch_size", type=int, default=1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--apply_confidence_mask", dest="apply_confidence_mask", action="store_true", default=True)
    parser.add_argument("--no-apply_confidence_mask", dest="apply_confidence_mask", action="store_false")
    parser.add_argument("--confidence_percentile", type=float, default=15.0)
    parser.add_argument("--use_multiview_confidence", dest="use_multiview_confidence", action="store_true", default=True)
    parser.add_argument("--no-use_multiview_confidence", dest="use_multiview_confidence", action="store_false")
    parser.add_argument("--mv_depth_abs_thresh", type=float, default=0.02)
    parser.add_argument("--mv_depth_rel_thresh", type=float, default=0.02)
    parser.add_argument("--edge_normal_threshold", type=float, default=5.0)
    parser.add_argument("--edge_depth_threshold", type=float, default=0.03)

    parser.add_argument("--min_map_conf", type=float, default=0.08)
    parser.add_argument("--min_region_area", type=int, default=48)
    parser.add_argument("--cluster_split_area", type=int, default=1800)
    parser.add_argument("--max_region_clusters", type=int, default=16)
    parser.add_argument("--max_points_per_region", type=int, default=3000)
    parser.add_argument("--min_points_per_projection", type=int, default=24)
    parser.add_argument("--min_projection_support", type=int, default=2)
    parser.add_argument("--min_views_for_geometry", type=int, default=2)
    parser.add_argument("--depth_abs_tol", type=float, default=0.015)
    parser.add_argument("--depth_rel_tol", type=float, default=0.025)
    parser.add_argument("--static_depth_rel_tol", type=float, default=0.035)
    parser.add_argument("--static_rgb_tol", type=float, default=0.12)
    parser.add_argument("--geometry_weight", type=float, default=0.62)
    parser.add_argument("--prior_weight", type=float, default=0.30)
    parser.add_argument("--conf_weight", type=float, default=0.08)
    parser.add_argument("--keep_score", type=float, default=0.38)
    parser.add_argument("--close_kernel", type=int, default=5)
    parser.add_argument("--close_iters", type=int, default=1)
    parser.add_argument("--open_kernel", type=int, default=3)
    parser.add_argument("--dilate_kernel", type=int, default=3)
    parser.add_argument("--dilate_iters", type=int, default=1)
    parser.add_argument("--keep_largest_components", type=int, default=0)
    parser.add_argument("--edge_refine", dest="edge_refine", action="store_true", default=True)
    parser.add_argument("--no-edge_refine", dest="edge_refine", action="store_false")
    parser.add_argument("--edge_refine_prior_floor", type=float, default=0.28)
    parser.add_argument("--edge_refine_otsu_scale", type=float, default=0.72)
    parser.add_argument("--edge_refine_sure_fg_score", type=float, default=0.62)
    parser.add_argument("--edge_refine_bg_score", type=float, default=0.12)
    parser.add_argument("--edge_refine_component_min_score", type=float, default=0.22)
    parser.add_argument("--edge_refine_min_area", type=int, default=36)
    parser.add_argument("--edge_refine_close_kernel", type=int, default=5)
    parser.add_argument("--edge_refine_fill_hole_area", type=int, default=900)
    parser.add_argument("--grabcut_band_kernel", type=int, default=21)
    parser.add_argument("--grabcut_band_iters", type=int, default=1)
    parser.add_argument("--grabcut_sure_fg_kernel", type=int, default=7)
    parser.add_argument("--grabcut_iters", type=int, default=2)
    parser.add_argument("--prior_rescue_score", type=float, default=0.55)
    parser.add_argument("--prior_min_area", type=int, default=48)
    parser.add_argument("--prior_component_min_score", type=float, default=0.18)
    parser.add_argument("--prior_close_kernel", type=int, default=5)
    parser.add_argument("--prior_fill_hole_area", type=int, default=500)
    parser.add_argument("--guard_min_iou_with_prior", type=float, default=0.82)
    parser.add_argument("--guard_min_area_ratio", type=float, default=0.82)
    parser.add_argument("--guard_max_area_ratio", type=float, default=1.18)
    parser.add_argument("--guard_min_boundary_gain", type=float, default=0.01)
    parser.add_argument("--sam2_box_pad", type=int, default=12)
    parser.add_argument("--sam2_num_positive_points", type=int, default=6)
    parser.add_argument("--sam2_num_negative_points", type=int, default=8)
    parser.add_argument("--sam2_min_area", type=int, default=48)
    parser.add_argument("--sam2_min_iou_with_prior", type=float, default=0.78)
    parser.add_argument("--sam2_min_area_ratio", type=float, default=0.80)
    parser.add_argument("--sam2_max_area_ratio", type=float, default=1.20)
    parser.add_argument("--sam2_edge_gain_weight", type=float, default=0.15)
    parser.add_argument("--sam2_iou_weight", type=float, default=0.35)
    parser.add_argument("--final_smooth_close_kernel", type=int, default=9)
    parser.add_argument("--final_smooth_close_iters", type=int, default=1)
    parser.add_argument("--final_smooth_open_kernel", type=int, default=5)
    parser.add_argument("--final_smooth_open_iters", type=int, default=1)
    parser.add_argument("--final_smooth_blur_kernel", type=int, default=9)
    parser.add_argument("--final_smooth_threshold", type=float, default=127.0)
    parser.add_argument("--final_smooth_fill_hole_area", type=int, default=1200)
    parser.add_argument("--final_poly_epsilon_ratio", type=float, default=0.006)
    parser.add_argument("--final_poly_min_area", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    input_dir = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = get_device()
    print(f"Using device: {device}")
    print("Loading VGGT4D...")
    vggt4d_model = load_vggt4d_model(args.vggt_ckpt, device)
    print("Loading MapAnything...")
    mapanything_model = load_mapanything_model(args, device)
    sam2_predictor = None
    if args.sam2_refine:
        print("Loading SAM2...")
        sam2_predictor = load_sam2_predictor(args, device)

    scene_dirs = resolve_scene_dirs(input_dir)
    if not scene_dirs:
        raise ValueError(f"No image scenes found under {input_dir}")
    print(f"Found {len(scene_dirs)} scene(s)")

    for scene_dir in scene_dirs:
        scene_out = output_root / scene_dir.name
        process_scene(scene_dir, scene_out, vggt4d_model, mapanything_model, sam2_predictor, device, args)

    print(f"\nAll done. Results saved to {output_root}")


if __name__ == "__main__":
    main()
