import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def run_sd_inpainting(image_path, mask_path, prompt, output_path, negative_prompt="artifacts, blur, shadow, ghosting, moving object, distortion"):
    """
    Uses Stable Diffusion to generatively inpaint a single keyframe, 
    with Pre-Generation Dilation and Gaussian Feathering for seamless blending.
    """
    print(f"[Stable Diffusion] Loading Inpainting Pipeline...")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.enable_attention_slicing()

    # ==========================================
    # [CORE FIX 1] Pre-Generation Moderate Dilation
    # ==========================================
    # Read mask with OpenCV first
    mask_cv2_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Dilate the mask moderately (15x15) so SD doesn't see motion blur residues.
    kernel_sd = np.ones((15, 15), np.uint8)
    mask_cv2_dilated = cv2.dilate(mask_cv2_raw, kernel_sd, iterations=1)
    
    # Convert back to PIL for diffusers
    mask_image = Image.fromarray(mask_cv2_dilated).convert("L")
    init_image = Image.open(image_path).convert("RGB")

    # Spatial Resolution Alignment
    orig_w, orig_h = init_image.size
    target_w = (orig_w // 8) * 8
    target_h = (orig_h // 8) * 8

    print(f"[Stable Diffusion] Generating background at {target_w}x{target_h} using prompt: '{prompt}'...")
    
    # Passed in a generic negative_prompt to prevent hallucinations without breaking generalization
    result_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        width=target_w,
        height=target_h,
        num_inference_steps=50,
        strength=1.0, 
    ).images[0]

    # Resize back to original
    result_image = result_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    # ==========================================
    # [CORE FIX 2] Soft Feathering Blending (The Game Changer)
    # ==========================================
    sd_cv2 = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
    orig_cv2 = cv2.imread(image_path)

    # Apply Gaussian Blur to the dilated mask to create a soft transition (Feathering)
    # This prevents the "hard patch" effect that causes ProPainter to crash or tear the video
    blur_kernel_size = (21, 21)
    mask_blurred = cv2.GaussianBlur(mask_cv2_dilated, blur_kernel_size, 0)

    # Normalize mask to 0.0 ~ 1.0 for Alpha Blending
    mask_norm = mask_blurred.astype(np.float32) / 255.0
    mask_norm = np.expand_dims(mask_norm, axis=-1)

    # Blend: Softly mix the SD generated background with the original image
    blended_cv2 = sd_cv2 * mask_norm + orig_cv2 * (1.0 - mask_norm)
    blended_cv2 = blended_cv2.astype(np.uint8)
    
    cv2.imwrite(output_path, blended_cv2)
    print(f"[Stable Diffusion] Keyframe seamlessly blended and saved to {output_path}")
    
    # Free VRAM
    del pipe
    torch.cuda.empty_cache()


def get_auto_keyframe_indices(total_frames, n_keyframes=3):
    """
    Automatically calculates keyframe indices using uniform sampling.
    Example: total_frames=70, n_keyframes=3 -> [0, 34, 69]
    """
    if n_keyframes <= 0:
        return []
    if n_keyframes == 1:
        return [0]
    
    # Generate n_keyframes indices spread evenly across the video
    indices = np.linspace(0, total_frames - 1, n_keyframes, dtype=int)
    return indices.tolist()

