import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def collect_frames(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def load_frames(frame_paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read frame: {path}")
        frames.append(frame)
    return frames


def extract_yolo_masks(model: YOLO, frames: list[np.ndarray], target_classes: list[int]) -> list[np.ndarray]:
    raw_masks = []
    print("Running YOLO instance segmentation...")
    for frame in frames:
        result = model(frame, verbose=False)[0]
        height, width = frame.shape[:2]
        frame_mask = np.zeros((height, width), dtype=np.uint8)

        if result.masks is not None and result.boxes is not None:
            for mask_tensor, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) not in target_classes:
                    continue
                mask = cv2.resize(mask_tensor.cpu().numpy(), (width, height))
                mask = (mask > 0.5).astype(np.uint8) * 255
                frame_mask = cv2.bitwise_or(frame_mask, mask)

        raw_masks.append(frame_mask)
    return raw_masks


def is_moving(prev_img: np.ndarray | None, curr_img: np.ndarray, mask: np.ndarray, threshold: float) -> bool:
    if prev_img is None or np.sum(mask) == 0:
        return False

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    points_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=mask,
        maxCorners=50,
        qualityLevel=0.3,
        minDistance=7,
    )
    if points_prev is None:
        return False

    points_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points_prev, None)
    if points_curr is None or status is None:
        return False

    valid_prev = points_prev[status == 1]
    valid_curr = points_curr[status == 1]
    if len(valid_curr) == 0:
        return False

    movement = np.mean(np.linalg.norm(valid_curr - valid_prev, axis=1))
    return movement > threshold


def filter_dynamic_masks(
    frames: list[np.ndarray],
    raw_masks: list[np.ndarray],
    motion_threshold: float,
    dilation_kernel: int,
    dilation_iters: int,
) -> list[np.ndarray]:
    print("Filtering dynamic objects with sparse optical flow...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    dynamic_masks = []
    for idx, frame in enumerate(frames):
        prev_frame = frames[idx - 1] if idx > 0 else None
        if is_moving(prev_frame, frame, raw_masks[idx], motion_threshold):
            dynamic_masks.append(cv2.dilate(raw_masks[idx], kernel, iterations=dilation_iters))
        else:
            dynamic_masks.append(np.zeros_like(raw_masks[idx]))
    return dynamic_masks


def restore_frames(
    frames: list[np.ndarray],
    raw_masks: list[np.ndarray],
    dynamic_masks: list[np.ndarray],
    edge_kernel: int,
    temporal_min_offset: int,
    temporal_max_offset: int,
    inpaint_radius: int,
) -> list[np.ndarray]:
    print("Restoring backgrounds with temporal propagation and Telea fallback...")
    edge_kernel_mat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_kernel, edge_kernel))
    search_offsets = list(range(temporal_min_offset, temporal_max_offset + 1))
    if temporal_min_offset > 1:
        search_offsets += list(range(1, temporal_min_offset))

    restored_frames = []
    for idx, frame in enumerate(frames):
        restored = frame.copy()
        target_mask = cv2.dilate(dynamic_masks[idx], edge_kernel_mat, iterations=1)
        working_mask = target_mask.copy()

        for offset in search_offsets:
            if not np.any(working_mask > 0):
                break
            for direction in (-1, 1):
                neighbor_idx = idx + offset * direction
                if neighbor_idx < 0 or neighbor_idx >= len(frames):
                    continue

                clean_background = cv2.dilate(raw_masks[neighbor_idx], edge_kernel_mat, iterations=1) == 0
                transfer_area = clean_background & (working_mask > 0)
                restored[transfer_area] = frames[neighbor_idx][transfer_area]
                working_mask[transfer_area] = 0

        if np.any(working_mask > 0):
            restored = cv2.inpaint(restored, working_mask, inpaint_radius, cv2.INPAINT_TELEA)
        restored_frames.append(restored)

    return restored_frames


def save_images(images: list[np.ndarray], frame_paths: list[Path], output_dir: Path, suffix: str = ".png") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, frame_path in zip(images, frame_paths):
        cv2.imwrite(str(output_dir / f"{frame_path.stem}{suffix}"), image)


def save_video(frames: list[np.ndarray], output_video: Path, fps: float) -> None:
    if not frames:
        return
    output_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1 hand-crafted baseline: YOLO masks + optical flow + temporal inpainting.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root for masks and restored frames.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt", help="YOLO segmentation model path or Ultralytics model name.")
    parser.add_argument("--target_classes", type=parse_classes, default=[0, 32], help="Comma-separated COCO class ids, e.g. 0,32.")
    parser.add_argument("--motion_threshold", type=float, default=1.0, help="Minimum average optical-flow displacement for dynamic objects.")
    parser.add_argument("--dilation_kernel", type=int, default=7, help="Kernel size for dynamic-mask dilation.")
    parser.add_argument("--dilation_iters", type=int, default=1, help="Dilation iterations for dynamic masks.")
    parser.add_argument("--edge_kernel", type=int, default=15, help="Extra dilation kernel for inpainting boundary coverage.")
    parser.add_argument("--temporal_min_offset", type=int, default=10, help="Start temporal search from this frame offset.")
    parser.add_argument("--temporal_max_offset", type=int, default=30, help="Largest temporal search offset.")
    parser.add_argument("--inpaint_radius", type=int, default=5, help="Telea inpainting radius for remaining holes.")
    parser.add_argument("--output_video", type=Path, default=None, help="Optional mp4 output path for restored frames.")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS for --output_video.")
    args = parser.parse_args()

    frame_paths = collect_frames(args.image_dir)
    if not frame_paths:
        raise ValueError(f"No image frames found in {args.image_dir}")

    frames = load_frames(frame_paths)
    print(f"Loaded {len(frames)} frames from {args.image_dir}")

    model = YOLO(args.yolo_model)
    raw_masks = extract_yolo_masks(model, frames, args.target_classes)
    dynamic_masks = filter_dynamic_masks(
        frames,
        raw_masks,
        args.motion_threshold,
        args.dilation_kernel,
        args.dilation_iters,
    )
    restored_frames = restore_frames(
        frames,
        raw_masks,
        dynamic_masks,
        args.edge_kernel,
        args.temporal_min_offset,
        args.temporal_max_offset,
        args.inpaint_radius,
    )

    save_images(raw_masks, frame_paths, args.output_dir / "raw_masks")
    save_images(dynamic_masks, frame_paths, args.output_dir / "dynamic_masks")
    save_images(restored_frames, frame_paths, args.output_dir / "restored_frames", suffix=".jpg")

    if args.output_video is not None:
        save_video(restored_frames, args.output_video, args.fps)

    print(f"Done. Results saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
