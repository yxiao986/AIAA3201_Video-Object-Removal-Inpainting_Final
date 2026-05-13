import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) between a predicted mask and a ground truth mask.
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask (H, W).
        gt_mask (np.ndarray): Ground truth binary mask (H, W).
        
    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return intersection / union

def evaluate_mask_quality(pred_masks, gt_masks, threshold=0.5):
    """
    Calculates the Mean IoU (J_M) and IoU Recall (J_R) for a video sequence.
    Reference: VGGT4D paper evaluation metrics.
    
    Args:
        pred_masks (list or np.ndarray): Sequence of predicted masks.
        gt_masks (list or np.ndarray): Sequence of ground truth masks.
        threshold (float): Threshold for a successful segmentation to count towards recall.
        
    Returns:
        dict: A dictionary containing J_M (mean IoU) and J_R (recall).
    """
    assert len(pred_masks) == len(gt_masks), "Number of predicted masks and GT masks must match."
    
    iou_scores = [calculate_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    
    # J_M: Mean IoU across all frames
    j_mean = np.mean(iou_scores)
    
    # J_R: Recall (percentage of frames with IoU > threshold)
    j_recall = np.sum(np.array(iou_scores) > threshold) / len(iou_scores)
    
    return {
        "J_M": float(j_mean),
        "J_R": float(j_recall)
    }

def evaluate_video_quality(pred_frames, gt_frames):
    """
    Calculates the average PSNR and SSIM for an inpainted video sequence compared to clean GT frames.
    Note: Only applicable if clean background Ground Truth frames exist.
    
    Args:
        pred_frames (list of np.ndarray): Sequence of inpainted frames (BGR or RGB).
        gt_frames (list of np.ndarray): Sequence of clean ground truth frames (BGR or RGB).
        
    Returns:
        dict: A dictionary containing average PSNR and average SSIM.
    """
    assert len(pred_frames) == len(gt_frames), "Number of predicted frames and GT frames must match."
    
    psnr_scores = []
    ssim_scores = []
    
    for pred, gt in zip(pred_frames, gt_frames):
        # skimage expects data range to be specified if not explicitly inferable
        p_score = psnr(gt, pred, data_range=255)
        # ssim requires channel_axis to be specified for multichannel (color) images
        s_score = ssim(gt, pred, data_range=255, channel_axis=-1)
        
        psnr_scores.append(p_score)
        ssim_scores.append(s_score)
        
    return {
        "PSNR": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores))
    }