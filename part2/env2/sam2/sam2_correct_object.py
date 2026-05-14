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


def image_files(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 + periodic YOLO correction for multi-object dynamic masks.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output binary masks.")
    parser.add_argument("--sam2_repo_root", type=Path, default=None, help="Optional path to a local SAM2 repository.")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint.")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--yolo_model", type=str, default="yolov8m.pt", help="YOLO model path or Ultralytics model name.")
    parser.add_argument("--target_classes", type=parse_classes, default=[0, 32, 38], help="Comma-separated COCO class ids.")
    parser.add_argument("--prompt_interval", type=int, default=15, help="Run YOLO correction every N frames.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="SAM2 mask-logit threshold.")
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
    print("Loading SAM2 and YOLO models...")
    sam2_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint, device=device)
    yolo_model = YOLO(args.yolo_model)

    inference_state = sam2_predictor.init_state(video_path=str(args.image_dir))
    print(f"Injecting YOLO prompts every {args.prompt_interval} frame(s)...")

    for frame_idx in range(0, len(frames), args.prompt_interval):
        img = cv2.imread(str(frames[frame_idx]))
        if img is None:
            continue
        height, width = img.shape[:2]
        yolo_results = yolo_model(img, verbose=False)[0]

        if yolo_results.boxes is None:
            continue

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        positive_points = []

        for box, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls):
            cls_id = int(cls)
            if cls_id not in args.target_classes:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            min_x, min_y = min(min_x, x1), min(min_y, y1)
            max_x, max_y = max(max_x, x2), max(max_y, y2)
            if cls_id in [32, 38]:
                positive_points.append([(x1 + x2) / 2, (y1 + y2) / 2])

        if min_x == float("inf"):
            continue

        super_box = np.array([
            max(0, min_x - 15),
            max(0, min_y - 5),
            min(width, max_x + 60),
            min(height, max_y + (max_y - min_y) * 0.35),
        ])
        positive_points.append([(min_x + max_x) / 2, max_y + 10])
        points = np.array(positive_points, dtype=np.float32)
        labels = np.ones(len(positive_points), dtype=np.int32)

        sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            box=super_box,
            points=points,
            labels=labels,
        )
        print(f"  frame {frame_idx:04d}: added super-box + {len(positive_points)} positive point(s)")

    first_image = cv2.imread(str(frames[0]))
    height, width = first_image.shape[:2]
    video_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(len(frames))]

    print("Propagating corrected masks...")
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
