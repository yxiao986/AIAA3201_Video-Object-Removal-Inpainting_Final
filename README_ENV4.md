# Environment env4: VGGT4D and MapAnything

Use this environment for Part 3 experiments that replace VGGT4D geometry with MapAnything geometry.

## Installation

```bash
conda create -n env4 python=3.12 -y
conda activate env4

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy scikit-image scikit-learn einops tqdm matplotlib

cd external/map-anything
pip install -e .
cd ../..

cd external/VGGT4D
pip install -r requirements.txt
mkdir -p ckpts
wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" \
  -O ckpts/model_tracker_fixed_e20.pt
cd ../..
```

For the SAM2-refined MapAnything script, install SAM2 in the same env:

```bash
cd external/sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../../..
```

## VGGT4D + MapAnything Basic

```bash
conda activate env4
PYTHONPATH=external/VGGT4D:external/map-anything python part3/env4/map-anything/map_vggt4d_basic.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/mapanything_basic \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --mapanything_model facebook/map-anything \
  --max_frames 20
```

## VGGT4D + MapAnything + SAM2 Refinement

```bash
PYTHONPATH=external/VGGT4D:external/map-anything:external/sam2 python part3/env4/map-anything/map_vggt4d_sam2.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/mapanything_sam2 \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --mapanything_model_name facebook/map-anything \
  --sam2_refine \
  --sam2_repo_root external/sam2 \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --max_frames 20
```

This script has many tuning flags for confidence masking, geometry voting, GrabCut/SAM2 refinement, and final smoothing. Run `--help` for the full list.

If HuggingFace access is slow, add:

```bash
--hf_endpoint https://hf-mirror.com
```
