# Video Object Removal and Inpainting Code Overview

This folder contains the project-specific scripts modified for Project 3. It does not vendor the full upstream repositories. To reproduce the experiments, clone the required upstream repositories first, then run the scripts in the matching environment.

## README Files

```text
README_YY.md      Project structure, data layout, upstream repos, and upload notes.
README_ENV2.md    Part 1 baseline, Part 2 SAM2/Cutie masks, ProPainter, and mask evaluation.
README_ENV3.md    Part 3 VGGT4D and Pi3/Pi3X experiments.
README_ENV4.md    Part 3 VGGT4D and MapAnything experiments.
```

## Folder Structure

```text
code/
├── part1/
│   └── yolo_mask.py
├── part2/
│   └── env2/
│       ├── cutie/
│       │   └── cutie.py
│       └── sam2/
│           ├── sam2_basic.py
│           ├── sam2_correct_object.py
│           ├── sam2_expand.py
│           └── sam2_init_gt_expand.py
├── part3/
│   ├── env3/
│   │   ├── VGGT4D/
│   │   │   └── vggt4d_basic.py
│   │   └── pi3/
│   │       ├── pi3_vggt4d_basic.py
│   │       └── pi3_vggt4d_hybrid.py
│   └── env4/
│       └── map-anything/
│           ├── map_vggt4d_basic.py
│           └── map_vggt4d_sam2.py
└── utils/
    └── env2/
        └── eval_mask.py
```

The files named `*_readme.md` in this folder are copied upstream README notes used during development. They are not required for the final GitHub submission and can be removed before upload.

## Data Layout

All scripts accept input/output paths from the terminal. A convenient local layout is:

```text
data/
├── bmx-trees/
│   ├── 00000.jpg
│   └── ...
├── bmx-trees_mask/
│   ├── 00000.png
│   └── ...
├── tennis/
├── tennis_mask/
└── wild/

checkpoints/
├── sam2.1_hiera_large.pt
├── model_tracker_fixed_e20.pt
└── yolov8n-seg.pt

outputs/
├── part1/
├── masks/
├── propainter/
└── part3/
```

For VGGT4D/Pi3/MapAnything scripts, `data/scenes` should contain one folder per scene:

```text
data/scenes/
├── bmx-trees/
│   ├── 00000.jpg
│   └── ...
└── tennis/
    ├── 00000.jpg
    └── ...
```

## Upstream Repositories

Clone the upstream repositories outside or beside this project:

```bash
mkdir -p external

git clone https://github.com/facebookresearch/sam2.git external/sam2
git clone https://github.com/hkchengrex/Cutie.git external/Cutie
git clone https://github.com/sczhou/ProPainter.git external/ProPainter
git clone https://github.com/3DAgentWorld/VGGT4D.git external/VGGT4D
git clone https://github.com/yyfz/Pi3.git external/Pi3
git clone https://github.com/facebookresearch/map-anything.git external/map-anything
```

If an upstream URL changes, use the official repository from the corresponding paper/project page.

## Environment Map

```text
env2
├── part1/yolo_mask.py
├── part2/env2/sam2/*.py
├── part2/env2/cutie/cutie.py
├── external/ProPainter/inference_propainter.py
└── utils/env2/eval_mask.py

env3
├── part3/env3/VGGT4D/vggt4d_basic.py
├── part3/env3/pi3/pi3_vggt4d_basic.py
└── part3/env3/pi3/pi3_vggt4d_hybrid.py

env4
├── part3/env4/map-anything/map_vggt4d_basic.py
└── part3/env4/map-anything/map_vggt4d_sam2.py
```

## Notes for GitHub Upload

- Keep this repository as project glue code, not a copy of all upstream repositories.
- Do not commit checkpoints such as `*.pt`, `*.pth`, or `*.safetensors`.
- Do not commit raw datasets, generated masks, generated videos, or output folders.
- Put download links and exact run commands in the environment-specific README files.
- Before submission, remove copied upstream README files if they are only local notes.
