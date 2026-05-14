# Environment env2: Part 1, SAM2, Cutie, YOLO, ProPainter

Use this environment for the Part 1 hand-crafted baseline, Part 2 mask generation, Cutie propagation, mask evaluation, and ProPainter inpainting.

## Installation

```bash
conda create -n env2 python=3.10 -y
conda activate env2

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy pillow tqdm

cd external/sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../../..

cd external/Cutie
pip install -e .
python cutie/utils/download_models.py
cd ../..

cd external/ProPainter
pip install -r requirements.txt
cd ../..
```

Download the YOLO checkpoint manually if you want to use a local file:

```bash
mkdir -p checkpoints
# Put yolov8n-seg.pt, yolov8n.pt, or yolov8m.pt under checkpoints/
```

Pass local YOLO weights explicitly:

```bash
--yolo_model checkpoints/yolov8n-seg.pt
```

## Part 1: Hand-crafted YOLO + Optical Flow Baseline

`part1/yolo_mask.py` implements the baseline required by the project guidance:

- YOLOv8 segmentation extracts candidate object masks.
- Sparse Lucas-Kanade optical flow filters static objects and keeps moving objects.
- Morphological dilation expands masks to cover motion blur and object edges.
- Temporal background propagation borrows clean pixels from nearby frames.
- OpenCV Telea inpainting fills any remaining holes.

Example for the tennis sequence:

```bash
conda activate env2
python part1/yolo_mask.py \
  --image_dir data/tennis \
  --output_dir outputs/part1/tennis \
  --yolo_model checkpoints/yolov8n-seg.pt \
  --target_classes 0,32 \
  --motion_threshold 1.0 \
  --dilation_kernel 7 \
  --edge_kernel 15 \
  --output_video outputs/part1/tennis_restored.mp4 \
  --fps 24
```

Example for a bicycle/car-style scene:

```bash
python part1/yolo_mask.py \
  --image_dir data/bmx-trees \
  --output_dir outputs/part1/bmx-trees \
  --yolo_model checkpoints/yolov8n-seg.pt \
  --target_classes 1,2 \
  --motion_threshold 1.5 \
  --dilation_kernel 7 \
  --output_video outputs/part1/bmx-trees_restored.mp4
```

The script writes:

```text
outputs/part1/<scene>/
├── raw_masks/
├── dynamic_masks/
└── restored_frames/
```

## SAM2 First-frame YOLO Initialization

Use this when the first frame has reliable YOLO detections.

```bash
conda activate env2
python part2/env2/sam2/sam2_basic.py \
  --image_dir data/bmx-trees \
  --output_dir outputs/masks/bmx-trees_sam2_basic \
  --sam2_repo_root external/sam2 \
  --sam2_checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --yolo_model checkpoints/yolov8n.pt \
  --target_classes 2
```

## SAM2 Periodic YOLO Correction

Use this for scenes where objects drift, disappear, or need repeated correction.

```bash
python part2/env2/sam2/sam2_correct_object.py \
  --image_dir data/tennis \
  --output_dir outputs/masks/tennis_sam2_corrected \
  --sam2_repo_root external/sam2 \
  --sam2_checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --yolo_model checkpoints/yolov8m.pt \
  --target_classes 0,32,38 \
  --prompt_interval 15
```

## SAM2 with Dilated Masks

Use this when the inpainting model needs masks that cover motion blur or object boundaries.

```bash
python part2/env2/sam2/sam2_expand.py \
  --image_dir data/bmx-trees \
  --output_dir outputs/masks/bmx-trees_sam2_expand \
  --sam2_repo_root external/sam2 \
  --sam2_checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --yolo_model checkpoints/yolov8m.pt \
  --target_classes 2 \
  --dilation_kernel 7 \
  --dilation_iters 1
```

`sam2_init_gt_expand.py` currently calls the same implementation as `sam2_expand.py`; keep it only if you want a separate experiment entry name.

## Cutie Mask Propagation

Use this when you already have the first-frame mask and want to propagate it through the video.

```bash
python part2/env2/cutie/cutie.py \
  --image_dir data/tennis \
  --first_mask data/tennis_mask/00000.png \
  --output_dir outputs/masks/tennis_cutie \
  --max_internal_size 480 \
  --output_ext .png
```

## ProPainter Inpainting

Run ProPainter from the upstream repository. Put its weights under `external/ProPainter/weights/` or let the official script download them during the first run.

```bash
cd external/ProPainter
python inference_propainter.py \
  --video ../../data/bmx-trees \
  --mask ../../outputs/masks/bmx-trees_sam2_expand \
  --output ../../outputs/propainter/bmx-trees \
  --fp16
cd ../..
```

## Mask Evaluation

Use `utils/env2/eval_mask.py` for IoU mean and IoU recall:

```bash
python utils/env2/eval_mask.py \
  --pred_dir outputs/part3/mapanything_sam2/bmx-trees \
  --gt_dir data/bmx-trees_mask \
  --num_frames 30 \
  --pred_pattern 'dynamic_mask_{idx:04d}.png' \
  --gt_pattern '{idx:05d}.png'
```
