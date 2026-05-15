# AIAA3201 Project 3: Video Object Removal & Inpainting (Integrated Repository)

Welcome to our integrated repository for Project 3!

This project explores the task of dynamic video object removal and background restoration, ranging from hand-crafted baseline algorithms to state-of-the-art (SOTA) video diffusion models.

## ⚠️ Important Execution Notice: Multi-Environment Setup

To ensure the highest level of reproducibility and completely avoid dependency conflicts (especially between traditional computer vision libraries and heavy generative AI models like DiffuEraser), our team has adopted a **Decoupled Environment Strategy**.

Instead of forcing all components into a single, unstable virtual environment, we have divided the execution workspaces based on the specific architectural requirements of each pipeline.

**DO NOT attempt to run the entire project using a single environment.** Please refer to the specific Environment Readme files (`README_ENV1.md`, `README_ENV2.md`, etc.) for exact setup and execution instructions.

## 📦 Data & Weights Download (Action Required)

Due to GitHub's file size limits, the `data` directory (containing the Tennis, BMX-Trees, and full DAVIS datasets) and the heavy pre-trained model weights are not included in this repository.

**👉 Please download the complete data and weights from our Google Drive:**
**[Data & Weights](https://drive.google.com/drive/folders/1gHMDOBe13MfARnnJaSH83mVpkj3Qwfoe?usp=sharing)**

*Instructions:* After downloading the `.zip` file from the link above, extract it and place the `data/` folder directly into the root directory of this project before running any scripts.

## 📂 Repository Structure

Our repository is organized to physically isolate the code and dependencies for different environments while sharing the same underlying data and third-party tools.

```text
PROJECT3_INTEGRATED/
├── data/                       # Shared datasets (download from Google Drive)
│   ├── bmx-trees/              # Image sequence for BMX dataset
│   ├── bmx-trees_mask/         # Ground truth masks for BMX
│   ├── DAVIS/                  # Full DAVIS benchmark dataset
│   ├── my_video/               # Self-captured wild video frames
│   ├── tennis/                 # Image sequence for Tennis dataset
│   ├── tennis_mask/            # Ground truth masks for Tennis
│   ├── .gitkeep                # Git placeholder to retain the folder structure
│   └── *.mp4                   # Converted video files (used as inputs for Track-Anything UI)
├── part1/                      # Baseline approaches
│   ├── env1/Mask-RCNN/         # Mask R-CNN + optical-flow baseline
│   └── env2/YOLOv8/            # YOLO + optical-flow baseline
├── part2/                      # SOTA tracking and mask generation
│   ├── env1/Track-Anything/    # Interactive and automated tracking pipelines
│   └── env2/
│       ├── cutie/              # Cutie mask propagation
│       └── sam2/               # SAM2 + YOLO prompt pipelines
├── part3/                      # Exploration
│   ├── env1/ProPainter_Explore/# ProPainter, SD2D, and DiffuEraser experiments
│   ├── env3/
│   │   ├── VGGT4D/             # VGGT4D dynamic masks
│   │   ├── pi3/                # VGGT4D + Pi3/Pi3X geometry
│   │   └── SAM3_VGGT4D/        # VGGT4D coarse masks refined with SAM3
│   └── env4/map-anything/      # VGGT4D + MapAnything experiments
├── external/                   # Upstream repositories cloned locally
├── outputs/                    # Generated masks, frames, and videos
├── results/                    # Evaluation and comparison outputs
├── utils/                      # Shared utility scripts
├── README.md                   # This master documentation
├── README_ENV1.md              # Env1 setup and execution guide
├── README_ENV2.md              # Env2 setup and execution guide
├── README_ENV3.md              # Env3 setup and execution guide
├── README_ENV4.md              # Env4 setup and execution guide
├── requirements_env1.txt
├── requirements_env2.txt
├── requirements_env3.txt
└── requirements_env4.txt
```

## Environment Guide

Use the matching README and requirements file for each environment:

```text
README_ENV1.md    Env1: Mask R-CNN baseline, Track-Anything, ProPainter, DiffuEraser.
README_ENV2.md    Env2: YOLO baseline, SAM2, Cutie, ProPainter, mask evaluation.
README_ENV3.md    Env3: VGGT4D, Pi3/Pi3X, SAM3 + VGGT4D.
README_ENV4.md    Env4: VGGT4D + MapAnything, optional SAM2 refinement.
```

Install the base dependencies for an environment with:

```bash
pip install -r requirements_env1.txt
pip install -r requirements_env2.txt
pip install -r requirements_env3.txt
pip install -r requirements_env4.txt
```

Some upstream repositories also require their own `pip install -r requirements.txt` or `pip install -e .`; those steps are listed in the corresponding `README_ENV*.md`.

## Upstream Repositories

Clone only the upstream repositories required by the environment you are running:

```bash
mkdir -p external

git clone https://github.com/gaomingqi/Track-Anything.git external/Track-Anything
git clone https://github.com/sczhou/ProPainter.git external/ProPainter
git clone https://github.com/lixiaowen-xw/diffueraser.git external/DiffuEraser
git clone https://github.com/facebookresearch/sam2.git external/sam2
git clone https://github.com/facebookresearch/sam3.git external/sam3
git clone https://github.com/hkchengrex/Cutie.git external/Cutie
git clone https://github.com/3DAgentWorld/VGGT4D.git external/VGGT4D
git clone https://github.com/yyfz/Pi3.git external/Pi3
git clone https://github.com/facebookresearch/map-anything.git external/map-anything
```

If an upstream URL changes, use the official repository from the corresponding paper or project page.

## Command-Line Interfaces

Project scripts are organized to receive runtime paths from the terminal. Input folders, output folders, checkpoints, external repository paths, devices, and evaluation masks can be passed through command-line flags. Run any executable script with `--help` to see all supported options.

Example commands:

```bash
python utils/extract_frames.py \
  --video_path data/my_video.mp4 \
  --output_dir data/my_video

python part2/env2/sam2/sam2_expand.py \
  --image_dir data/bmx-trees \
  --output_dir outputs/masks/bmx-trees_sam2_expand \
  --sam2_repo_root external/sam2 \
  --sam2_checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --yolo_model checkpoints/yolov8m.pt

PYTHONPATH=external/VGGT4D:external/sam3 python part3/env3/SAM3_VGGT4D/SAM3_VGGT4D_improve.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/sam3_vggt4d_improve \
  --vggt4d_dir external/VGGT4D \
  --propainter_dir external/ProPainter \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --sam3_ckpt external/sam3_ms/sam3.pt \
  --sam3_bpe external/sam3_ms/bpe_simple_vocab_16e6.txt.gz
```

## Notes for GitHub Upload

- Keep this repository as project glue code, not a copy of all upstream repositories.
- Do not commit raw datasets, generated masks, generated videos, or output folders.
- Do not commit checkpoints such as `*.pt`, `*.pth`, or `*.safetensors`.
- Put download links and exact run commands in the environment-specific README files.

## Results Summary

Quantitative and qualitative results are summarized in `REPORT.md`. Environment-specific READMEs contain the exact commands used to reproduce each pipeline.
