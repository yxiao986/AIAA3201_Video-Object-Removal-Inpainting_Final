import argparse
from pathlib import Path
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

# Pi3X geometry backbone.
from pi3.models.pi3x import Pi3X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_scene(scene_dir: Path, output_dir: Path, vggt_model, pi3x_model):
    image_paths = sorted(list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png")))
    if not image_paths:
        return
    print(f"\nProcessing scene: {scene_dir.name} ({len(image_paths)} images)")

    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    n_img, _, h_img, w_img = images.shape
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extract spatio-temporal dynamic priors from VGGT4D QK tracking.
    print("  [VGGT4D] Stage 1: Extracting Spatio-Temporal Dynamic Maps...")
    with torch.no_grad():
        predictions1, qk_dict, enc_feat, _ = inference(vggt_model, images)
    
    qk_dict = organize_qk_dict(qk_dict, n_img)
    dyn_maps = extract_dyn_map(qk_dict, images)
    
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
        # Closing fills small internal gaps while keeping the outer boundary controlled.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove isolated background noise with connected-component analysis.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        
        # Ignore label 0 because it is the background component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        final_m = np.zeros_like(closed_mask)
        if len(areas) > 0:
            max_area = np.max(areas)
            for i, area in enumerate(areas):
                # Keep large regions and small-but-meaningful objects such as balls.
                if area > max_area * 0.05 or area > 50:
                    final_m[labels == i + 1] = 1
                    
        # Fill holes inside kept components before geometry refinement.
        contours, _ = cv2.findContours(final_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_m, contours, -1, 1, thickness=cv2.FILLED)

        refined_raw_masks.append(final_m)

    # Convert cleaned masks to a tensor for geometry refinement.
    dyn_masks_tensor = torch.tensor(np.array(refined_raw_masks)).bool().to(device)


    # Release VGGT4D intermediate features before running the geometry backbone.
    del qk_dict, enc_feat, feat_map, upsampled_map
    torch.cuda.empty_cache()

    # Stage 2: use Pi3X for depth and camera-pose prediction.
    print("  [Pi3X] Stage 2: Extracting High-Fidelity Geometry...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        pi3x_results = pi3x_model(images.unsqueeze(0))

    pred_cam2world = pi3x_results['camera_poses'][0].detach().cpu().numpy()
    pred_depths = pi3x_results['local_points'][0][..., 2].detach().cpu().numpy()
    pred_conf = torch.sigmoid(pi3x_results['conf'][0]).squeeze(-1).detach().cpu().numpy()

    # Pi3X does not return intrinsics here, so use a simple centered pinhole prior.
    focal_length = max(h_img, w_img)
    intrinsic = np.array([
        [focal_length, 0, w_img / 2],
        [0, focal_length, h_img / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    pred_intrinsic = np.tile(intrinsic, (n_img, 1, 1))

    # Stage 3: refine dynamic masks with geometry consistency.
    print("  [Hybrid] Stage 3: Refining Masks via 3D Consistency...")
    refiner = RefineDynMask(
        images, 
        torch.tensor(pred_depths).to(device),
        dyn_masks_tensor,
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device
    )
    
    refined_mask = refiner.refine_masks()
    del refiner
    torch.cuda.empty_cache()

    # Save masks and geometry outputs in the VGGT4D result format.
    print(f"  Saving results to {output_dir}")
    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, pred_cam2world)
    save_dynamic_masks(output_dir, refined_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vggt_ckpt", type=str, required=True)
    parser.add_argument("--pi3_model", type=str, default="yyfz233/Pi3X")
    parser.add_argument("--hf_endpoint", type=str, default=None, help="Optional HuggingFace endpoint mirror.")
    args = parser.parse_args()
    if args.hf_endpoint:
        import os
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    print("Loading VGGT4D Tracker...")
    vggt_model = VGGTFor4D().to(device).eval()
    vggt_model.load_state_dict(torch.load(args.vggt_ckpt, map_location=device, weights_only=True))

    print("Loading Pi3X Model...")
    pi3x_model = Pi3X.from_pretrained(args.pi3_model).to(device).eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scenes = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    for scene in scenes:
        process_scene(scene, output_dir / scene.name, vggt_model, pi3x_model)
        
    print("\nAll scenes processed successfully.")

if __name__ == "__main__":
    main()
