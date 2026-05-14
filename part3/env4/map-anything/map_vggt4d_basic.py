import argparse
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange

# VGGT4D dynamic-mask modules.
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (adaptive_multiotsu_variance,
                                       cluster_attention_maps,
                                       extract_dyn_map)
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (save_depth, save_depth_conf,
                                save_dynamic_masks, save_intrinsic_txt,
                                save_rgb, save_tum_poses)
from vggt.utils.load_fn import load_and_preprocess_images

# MapAnything geometry backbone.
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_scene(scene_dir: Path, output_dir: Path, vggt_model, ma_model, max_frames: int):
    image_paths = sorted(list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png")))
    if not image_paths:
        return
        
    if max_frames > 0 and len(image_paths) > max_frames:
        print(f"Warning: {len(image_paths)} frames may exceed GPU memory.")
        print(f"Using the first {max_frames} frame(s) for this run.")
        image_paths = image_paths[:max_frames]

    print(f"\nProcessing scene: {scene_dir.name} ({len(image_paths)} images)")

    # VGGT4D uses its own preprocessing and normalization path.
    images_vggt = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    n_img, _, h_img, w_img = images_vggt.shape
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extract spatio-temporal dynamic priors from VGGT4D QK tracking.
    print("  [VGGT4D] Stage 1: Extracting Spatio-Temporal Dynamic Maps...")
    with torch.no_grad():
        predictions1, qk_dict, enc_feat, _ = inference(vggt_model, images_vggt)
    
    qk_dict = organize_qk_dict(qk_dict, n_img)
    dyn_maps = extract_dyn_map(qk_dict, images_vggt)
    
    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
    
    upsampled_map = F.interpolate(rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"), 
                                  size=(h_img, w_img), mode='bilinear', align_corners=False)
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")
    
    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    raw_dyn_masks = (upsampled_map > thres).cpu().numpy().astype(np.uint8)

    # Improve mask recall and precision with closing and connected-component filtering.
    print("  [Processing] Applying Advanced Edge-Aware Post-Processing...")
    refined_raw_masks = []
    
    for m in raw_dyn_masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        final_m = np.zeros_like(closed_mask)
        if len(areas) > 0:
            max_area = np.max(areas)
            for i, area in enumerate(areas):
                if area > max_area * 0.05 or area > 50:
                    final_m[labels == i + 1] = 1
                    
        contours, _ = cv2.findContours(final_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_m, contours, -1, 1, thickness=cv2.FILLED)
        refined_raw_masks.append(final_m)

    dyn_masks_tensor = torch.tensor(np.array(refined_raw_masks)).bool().to(device)

    # Release VGGT4D intermediate features before running the geometry backbone.
    del qk_dict, enc_feat, feat_map, upsampled_map
    torch.cuda.empty_cache()

    # Stage 2: use MapAnything for depth, camera poses, intrinsics, and confidence.
    print("  [MapAnything] Stage 2: Extracting High-Fidelity Geometry and Real Intrinsics...")
    
    # MapAnything expects DINOv2-normalized inputs at a patch-compatible resolution.
    image_str_paths = [str(p) for p in image_paths]
    views_ma = load_images(
        folder_or_list=image_str_paths,
        resolution_set=518,
        norm_type="dinov2",
        patch_size=14
    )

    # Use full precision here because this path was more stable than autocast.
    with torch.no_grad():
        ma_predictions = ma_model.infer(
            views_ma,
            memory_efficient_inference=True,
            use_amp=False,
        )

    # Extract and stack MapAnything predictions into VGGT4D-compatible arrays.
    ma_depths, ma_poses, ma_intrinsics, ma_confs = [], [], [], []
    # for pred in ma_predictions:
    #     ma_depths.append(pred["depth_z"].squeeze(-1))       # (H, W)
    #     ma_poses.append(pred["camera_poses"].squeeze(0))    # (4, 4)
    #     ma_intrinsics.append(pred["intrinsics"].squeeze(0)) # (3, 3)
    #     ma_confs.append(pred["conf"].squeeze(0))            # (H, W)
    
    for pred in ma_predictions:
        # Remove singleton dimensions introduced by the model API.
        ma_depths.append(pred["depth_z"].squeeze())       
        ma_poses.append(pred["camera_poses"].squeeze())    
        ma_intrinsics.append(pred["intrinsics"].squeeze()) 
        ma_confs.append(pred["conf"].squeeze())

    pred_depths_tensor = torch.stack(ma_depths).float()     # (n_img, H, W)
    pred_conf_tensor = torch.stack(ma_confs).float()        # (n_img, H, W)
    pred_cam2world = torch.stack(ma_poses).float().cpu().numpy()
    pred_intrinsic = torch.stack(ma_intrinsics).float().cpu().numpy()

    h_ma, w_ma = pred_depths_tensor.shape[1], pred_depths_tensor.shape[2]

    # Resize MapAnything depth/confidence maps to the VGGT4D image size.
    if (h_ma, w_ma) != (h_img, w_img):
        pred_depths_tensor = F.interpolate(pred_depths_tensor.unsqueeze(1), size=(h_img, w_img), mode='bilinear', align_corners=False).squeeze(1)
        pred_conf_tensor = F.interpolate(pred_conf_tensor.unsqueeze(1), size=(h_img, w_img), mode='bilinear', align_corners=False).squeeze(1)
        
        # Rescale intrinsics after resizing the geometry maps.
        scale_x = w_img / w_ma
        scale_y = h_img / h_ma
        for i in range(n_img):
            pred_intrinsic[i, 0, 0] *= scale_x
            pred_intrinsic[i, 1, 1] *= scale_y
            pred_intrinsic[i, 0, 2] *= scale_x
            pred_intrinsic[i, 1, 2] *= scale_y

    pred_depths = pred_depths_tensor.cpu().numpy()
    pred_conf = pred_conf_tensor.cpu().numpy()

    # Stage 3: refine dynamic masks with MapAnything geometry consistency.
    print("  [Hybrid] Stage 3: Refining Masks via 3D Consistency...")
    refiner = RefineDynMask(
        images_vggt, 
        torch.tensor(pred_depths).to(device),
        dyn_masks_tensor,
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).float().to(device), 
        device
    )
    
    refined_mask = refiner.refine_masks()
    del refiner
    torch.cuda.empty_cache()

    # Save masks and geometry outputs in the VGGT4D result format.
    print(f"  Saving results to {output_dir}")
    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images_vggt)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, pred_cam2world)
    save_dynamic_masks(output_dir, refined_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vggt_ckpt", type=str, required=True)
    parser.add_argument("--mapanything_model", type=str, default="facebook/map-anything")
    parser.add_argument("--max_frames", type=int, default=20, help="Only process first N frames; set <=0 for all frames.")
    parser.add_argument("--hf_endpoint", type=str, default=None, help="Optional HuggingFace endpoint mirror.")
    args = parser.parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    print("Loading VGGT4D Tracker...")
    vggt_model = VGGTFor4D().to(device).eval()
    vggt_model.load_state_dict(torch.load(args.vggt_ckpt, map_location=device, weights_only=True))

    print("Loading MapAnything Model...")
    # Reduce CUDA memory fragmentation for long multi-view sequences.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    ma_model = MapAnything.from_pretrained(args.mapanything_model).to(device).eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scenes = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    for scene in scenes:
        process_scene(scene, output_dir / scene.name, vggt_model, ma_model, args.max_frames)
        
    print("\nAll scenes processed successfully.")

if __name__ == "__main__":
    main()
