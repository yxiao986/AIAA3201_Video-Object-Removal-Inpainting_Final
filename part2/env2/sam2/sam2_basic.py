import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def expand_box(box: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([
        max(0, cx - bw / 2),
        max(0, cy - bh / 2),
        min(width, cx + bw / 2),
        min(height, cy + bh / 2),
    ])


def image_files(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize SAM2 video masks from YOLO detections on the first frame.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output binary masks.")
    parser.add_argument("--sam2_repo_root", type=Path, default=None, help="Optional path to a local SAM2 repository.")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint, e.g. checkpoints/sam2.1_hiera_large.pt.")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config name/path.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLO model path or Ultralytics model name.")
    parser.add_argument("--target_classes", type=parse_classes, default=[2], help="Comma-separated COCO class ids, e.g. 0,32,38.")
    parser.add_argument("--box_scale", type=float, default=1.3, help="Scale factor applied to YOLO boxes before prompting SAM2.")
    parser.add_argument("--mask_threshold", type=float, default=0.3, help="SAM2 mask-logit threshold.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.sam2_repo_root is not None:
        sys.path.insert(0, str(args.sam2_repo_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    frames = image_files(args.image_dir)
    if not frames:
        raise ValueError(f"No image frames found in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("Loading YOLO and SAM2 models...")
    yolo_model = YOLO(args.yolo_model)
    sam2_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint, device=device)

    inference_state = sam2_predictor.init_state(video_path=str(args.image_dir))

    print("Extracting initial prompts from frame 0...")
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")
    height, width = first_frame.shape[:2]

    prompts = []
    yolo_results = yolo_model(first_frame, verbose=False)[0]
    if yolo_results.boxes is not None:
        for box, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls):
            if int(cls) in args.target_classes:
                prompts.append(expand_box(box.cpu().numpy(), width, height, args.box_scale))

    if not prompts:
        raise ValueError("No target object was detected in frame 0; SAM2 cannot be initialized.")

    print(f"Found {len(prompts)} object(s); injecting prompts into SAM2...")
    for obj_id, bbox in enumerate(prompts, start=1):
        sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=bbox,
        )

    print("Propagating masks through the video...")
    video_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(len(frames))]
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        frame_mask = np.zeros((height, width), dtype=np.uint8)
        for i, _ in enumerate(out_obj_ids):
            mask = (out_mask_logits[i, 0].cpu().numpy() > args.mask_threshold).astype(np.uint8)
            frame_mask = cv2.bitwise_or(frame_mask, mask * 255)
        video_masks[out_frame_idx] = frame_mask

    for frame_path, mask in zip(frames, video_masks):
        cv2.imwrite(str(args.output_dir / f"{frame_path.stem}.png"), mask)

    print(f"Done. Masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
