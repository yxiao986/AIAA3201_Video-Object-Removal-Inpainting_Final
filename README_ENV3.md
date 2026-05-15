# Environment env3: VGGT4D, Pi3 and SAM3

Use this environment for Part 3 experiments based on VGGT4D, Pi3/Pi3X and SAM3.

## Installation

```bash
conda create -n env3 python=3.10 -y
conda activate env3

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy scikit-image scikit-learn einops tqdm matplotlib safetensors huggingface_hub

cd external/VGGT4D
pip install -r requirements.txt
mkdir -p ckpts
wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" \
  -O ckpts/model_tracker_fixed_e20.pt
cd ../..

cd external/Pi3
pip install -r requirements.txt
cd ../..

cd external/sam3
pip install -e .
cd ../..

# SAM3 model files
# This folder is used by SAM3_VGGT4D/SAM3_VGGT4D.py and SAM3_VGGT4D/SAM3_VGGT4D_improve.py
mkdir -p external/sam3_ms

# Login is required before downloading gated SAM3 model files
huggingface-cli login

huggingface-cli download facebook/sam3 \
  sam3.pt \
  bpe_simple_vocab_16e6.txt.gz \
  --local-dir external/sam3_ms \
  --local-dir-use-symlinks False
```
Before first use, request access to the SAM3 checkpoint on HuggingFace and login, if Sam3 does not pass on Huggingface, it can also be obtained on ModelScope.  

## VGGT4D Baseline Dynamic Masks

Copy or symlink `part3/env3/VGGT4D/vggt4d_basic.py` into the VGGT4D repository root, or run it with `PYTHONPATH` pointing to the VGGT4D repo:

```bash
conda activate env3
PYTHONPATH=external/VGGT4D python part3/env3/VGGT4D/vggt4d_basic.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/vggt4d_basic \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt
```

Input scene layout:

```text
data/scenes/
├── bmx-trees/
│   ├── 00000.jpg
│   └── ...
└── tennis/
    ├── 00000.jpg
    └── ...
```

## VGGT4D + Pi3X Geometry

Run from this repo with both VGGT4D and Pi3 on `PYTHONPATH`:

```bash
PYTHONPATH=external/VGGT4D:external/Pi3 python part3/env3/pi3/pi3_vggt4d_basic.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/pi3_basic \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --pi3_model yyfz233/Pi3X
```

The hybrid version adds stronger morphology and component filtering:

```bash
PYTHONPATH=external/VGGT4D:external/Pi3 python part3/env3/pi3/pi3_vggt4d_hybrid.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/pi3_hybrid \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --pi3_model yyfz233/Pi3X
```

If HuggingFace access is slow, add:

```bash
--hf_endpoint https://hf-mirror.com
```
## SAM3 + VGGT4D

```bash
PYTHONPATH=external/VGGT4D:external/sam3 python part3/env3/SAM3_VGGT4D/SAM3_VGGT4D.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/sam3_vggt4d \
  --vggt4d_dir external/VGGT4D \
  --propainter_dir external/ProPainter \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --sam3_ckpt external/sam3_ms/sam3.pt \
  --sam3_bpe external/sam3_ms/bpe_simple_vocab_16e6.txt.gz \
  --gt_bmx data/gt_masks/bmx-trees \
  --gt_tennis data/gt_masks/tennis \
  --scene both \
  --chunk_size 20
```

The improved version adds stronger morphology and component filtering:

```bash
PYTHONPATH=external/VGGT4D:external/sam3 python part3/env3/SAM3_VGGT4D/SAM3_VGGT4D_improve.py \
  --input_dir data/scenes \
  --output_dir outputs/part3/sam3_vggt4d_improve \
  --vggt4d_dir external/VGGT4D \
  --propainter_dir external/ProPainter \
  --vggt_ckpt external/VGGT4D/ckpts/model_tracker_fixed_e20.pt \
  --sam3_ckpt external/sam3_ms/sam3.pt \
  --sam3_bpe external/sam3_ms/bpe_simple_vocab_16e6.txt.gz \
  --gt_bmx data/gt_masks/bmx-trees \
  --gt_tennis data/gt_masks/tennis \
  --scene both \
  --chunk_size 20
```
