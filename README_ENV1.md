# Project 3: Video Object Removal & Inpainting

This repository documentation outlines the execution steps for **Environment 1 (Env1)**. It includes our baseline hand-crafted approach, interactive tracking (Track-Anything), and generative video inpainting explorations (ProPainter & DiffuEraser).

**⚠️ Important:** Please ensure you are running all commands from the **root directory** of the integrated project (`PROJECT3_INTEGRATED/`), NOT from within the subdirectories.

## Environment Setup

### 1. Conda Environment
```bash
# Create and activate a clean Conda environment for ENV1
conda create -n cv_proj3_env1 python=3.10 -y
conda activate cv_proj3_env1

# Install required packages
pip install -r requirements_env1.txt
```

### SOTA Models & Weights Setup
Since this project utilizes heavy models, you need to clone the official repositories into the `shared third_party` folder and download their respective pre-trained weights.

```bash
# From the project root, setup the shared third_party directory
mkdir third_party
cd third_party
git clone https://github.com/gaomingqi/Track-Anything.git
git clone https://github.com/sczhou/ProPainter.git
git clone https://github.com/lixiaowen-xw/diffueraser.git
cd ..
```
**Note**: Follow the instructions in their respective official repos to download the weights, or extract our pre-packaged [weights](https://drive.google.com/drive/folders/1gHMDOBe13MfARnnJaSH83mVpkj3Qwfoe?usp=sharing) directly into the third_party/ directory.

## Repository Structure (Env1 Focus)
```Plaintext
PROJECT3_INTEGRATED/
├── data/                       # Shared datasets directory
├── part1/
│   └── env1/
│       └── Mask-RCNN/          # Baseline Pipeline
│           ├── inpainter.py
│           ├── main.py
│           ├── mask_extractor.py
│           └── run_davis.py
├── part2/
│   └── env1/
│       └── Track-Anything/     # SOTA Orchestration
│           ├── launch_ui.py
│           ├── main.py
│           └── run_davis_subset.py
├── part3/
│   └── env1/
│       └── ProPainter_Explore/ # Generative Exploration
│           ├── main.py
│           └── run_davis.py
├── third_party/                # Shared SOTA repositories
├── utils/                      # Shared utility scripts
├── README_ENV1.md              # THIS DOCUMENT
└── requirements_env1.txt       # Dependencies for Env1                  # Output directory for generated artifacts
```
## Part 1: The Baseline Hand-crafted Approach

### Code Explanation
- `mask_extractor.py`: Handles the detection phase. It first retrieves soft masks from Mask R-CNN, thresholds them, and then compares frame $t-1$ and $t$ using optical flow to discard stationary bounding boxes.
- `inpainter.py`: Handles the restoration phase. It implements a temporal sliding window (default $\pm 15$ frames) to search for clean background pixels at the exact spatial location. If holes remain, cv2.inpaint patches them spatially.
- `main.py`: The entry point script. It uses argparse to dynamically load specific datasets, routes data through the extractor and inpainter, calculates evaluation metrics (if GT masks are provided), and saves the final frames and metrics.json.

### How to Run
Navigate to the project root directory before executing the commands.

#### 1. Tennis Dataset 
   
Runs the pipeline and evaluates mask accuracy against the provided Ground Truth.
```bash
python part1/env1/Mask-RCNN/main.py \
    --dataset_name tennis \
    --data_dir data/tennis \
    --gt_mask_dir data/tennis_mask \
    --output_base_dir results/part1/MaskRCNN
```

#### 2. BMX-Trees Dataset
```bash
python part1/env1/Mask-RCNN/main.py \
    --dataset_name bmx-trees \
    --data_dir data/bmx-trees \
    --gt_mask_dir data/bmx-trees_mask \
    --output_base_dir results/part1/MaskRCNN
```

#### 3. Wild Video Dataset (Mandatory)
For our self-captured video, there are no Ground Truth masks. 
```bash
python part1/env1/Mask-RCNN/main.py \
    --dataset_name my_video \
    --data_dir data/my_video \
    --output_base_dir results/part1/MaskRCNN
```

#### 4. DAVIS Dataset Full Evaluation (Batch Processing)
To rigorously evaluate the accuracy of our baseline extraction algorithm across the entire standard benchmark, we provide a batch processing script. This script automatically iterates through all available video sequences in the `data/DAVIS/JPEGImages/480p` directory, generates dynamic masks, compares them against the `Annotations/480p` directory, and computes the global average $\mathcal{J}_M$ and $\mathcal{J}_R$ metrics for the dataset.

```bash
# Run mask evaluation only
python part1/env1/Mask-RCNN/run_davis.py \
    --davis_root data/DAVIS \
    --output_dir results/part1/davis

# Run with spatial-temporal inpainting (Time-consuming)
python part1/env1/Mask-RCNN/run_davis.py \
    --davis_root data/DAVIS \
    --output_dir results/part1/davis \
    --run_inpainting
```

### Outputs & Artifacts
After running the commands, check the `../results/part1_baseline/[dataset_name]/` directory. You will find:
- `masks/`: The binary dynamic masks generated by our extractor.
- `inpainted/`: The final restored video frames.
- `metrics.json`: A JSON file containing the calculated evaluation metrics ($\mathcal{J}_M$ and $\mathcal{J}_R$) for the sequence (only generated if `--gt_mask_dir` was provided).
- 
## Part 2: SOTA Reproduction

This section utilizes foundation models to handle complex tracking scenarios (e.g., occlusion, motion blur) and performs high-fidelity video inpainting. 

To ensure the highest mask quality ($\mathcal{J}_M$ and $\mathcal{J}_R$), we employ an **Interactive-to-Automated Workflow**: User interaction is only required for the first frame via a Web UI, and the model automatically propagates the mask for the remaining frames.

### Code Architecture
- `main.py`: The main orchestration script that evaluates mask quality and executes the video inpainting engine.
- `third_party/Track-Anything/app.py`: The official Gradio UI used to obtain the initial high-precision prompt.
- `third_party/ProPainter/inference_propainter.py`: SOTA video inpainting engine.
- `utils/make_video.py`: script to convert the image sequence of a dataset into a video.

### How to Run

#### Step 0: Data Preparation (Image to Video)

**Note: If you downloaded data from our google drive, you can skipped this step as we already generate it for you.**

The Track-Anything UI requires a single video file (.mp4) for input. Use the provided script to convert the image sequence of a dataset into a video:

```bash
# For tennis dataset
python utils/make_video.py --input_dir data/tennis --output_file data/tennis.mp4

# For bmx-trees dataset
python utils/make_video.py --input_dir data/bmx-trees --output_file data/bmx-trees.mp4
```

#### Step 1: Interactive Mask Generation

Launch the UI using our safe launcher from the root directory:
```bash
python part2/env1/Track-Anything/launch_ui.py
```
1. Open your browser to the local URL provided in the terminal (usually `http://127.0.0.1:12212`).
2. Upload your target video frames (e.g., `data/tennis`.
3. Click on the target object (e.g., the tennis player) in the first frame
4. Click "Add new object", then click "Tracking".
5. Once tracking is 100% complete, manually move all the generated `.png` mask files from `third_party/Track-Anything/result/mask/tennis/` to your project's target directory: `results/part2/Track-Anything/tennis/masks`.

#### Step 2: Evaluation & Inpainting
Once the masks are saved, open a new terminal and run the main orchestration script:
```Bash
python part2/env1/Track-Anything/main.py \
    --dataset_name tennis \
    --data_dir data/tennis \
    --gt_mask_dir data/tennis_mask \
    --output_base_dir results/part2/Track-Anything
```
*The pipeline will automatically apply mask dilation, compute J_M and J_R metrics, and save the final inpainted video frames to `results/part2_sota/tennis/inpainted/`.*

### ⚠️ Mandatory Bug Fixes in Track-Anything (app.py)

Before running the Web UI, you **must** manually patch two critical bugs in the original author's `app.py` within the `third_party/Track-Anything/` directory to prevent immediate crashes on local machines.

#### Fix 1: Hardcoded CUDA Device (Invalid Device Ordinal Error)

The original author hardcoded the target GPU to their specific server setup (`cuda:3`). On single-GPU systems or machines without a 4th GPU, this causes an immediate `RuntimeError: CUDA error: invalid device ordinal`.
* **Action:** Open `third_party/Track-Anything/app.py` and **comment out or delete** the line where the device is forcefully overwritten (around line 381):
  ```python
  # args.device = "cuda:3"  <-- DELETE OR COMMENT OUT THIS LINE
    ```
#### Fix 2: Deprecated AV Package Crash (Video Generation)
The modern `torchvision.io.write_video` function in `app.py` relies on deprecated versions of the `av` package (`<10.0.0`), which no longer have pre-compiled binaries available on PyPI, leading to Cython compilation crashes on Windows.

To resolve this without altering the environment requirements, **we reverted to the author's original (commented out) OpenCV implementation** for video generation (around line 347 in `app.py`), with an added RGB-to-BGR color channel conversion to prevent color inversion:

```python
# Reverted and patched video generation function:
def generate_video_from_frames(frames, output_path, fps=30):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        bgr_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(bgr_frame)
    
    video.release()    
    return output_path
```

#### Fix 3: Force PNG Mask Export
By default, the original repository disables mask saving to save space, and even if enabled, exports them as `.npy` matrices instead of image files. To integrate seamlessly with our evaluation pipeline, we must force `.png` exportation.
* **Action:** Open `third_party/Track-Anything/app.py`. In line 270, insert:
```python
# ==========================================================
    mask_save_dir = os.path.join('./result/mask', video_state["video_name"].split('.')[0])
    os.makedirs(mask_save_dir, exist_ok=True)

    if "masks" in video_state and video_state["masks"] is not None:
        for i, mask in enumerate(video_state["masks"]):
            cv2.imwrite(os.path.join(mask_save_dir, f"{i:05d}.png"), (mask * 255).astype(np.uint8))
        print(f"\n Mask sequence has been saved to {mask_save_dir}\n")
    # ==========================================================


```


### Outputs & Artifacts
After running the commands, check the `../results/part2_sota/[dataset_name]/` directory. You will find:
- `masks/`: The binary dynamic masks generated by our extractor.
- `inpainted/`: The final restored video frames.
- `metrics.json`: A JSON file containing the calculated evaluation metrics ($\mathcal{J}_M$ and $\mathcal{J}_R$) for the sequence (only generated if `--gt_mask_dir` was provided).

### Advanced: End-to-End Interactive Evaluation on DAVIS Subset

The SOTA tracking pipeline (Track-Anything) is fundamentally designed as an **Interactive VOS (Video Object Segmentation)** framework. It requires a human-in-the-loop click to initialize zero-shot tracking.

To demonstrate its real-world effectiveness without the impracticality of manually clicking through all 50 DAVIS sequences, we have implemented a **Representative Subset Evaluation Strategy**. We selected highly challenging sequences characterized by heavy occlusion and fast motion (e.g., `dog`) to evaluate the interactive tracking precision ($\mathcal{J}_M$ and $\mathcal{J}_R$) and final inpainting quality.

#### Step-by-Step Reproduction Guide (Route A: Interactive Flow)

To reproduce our results on the challenging subset, please follow these exact steps to bridge the interactive UI and our automated evaluation script:

1. Prepare the video input:
   
**Note: If you downloaded data from our google drive, you can skipped this step as we already generate it for you.**
```bash
# Single Sequence Mode
python utils/make_video.py --input_folder data/DAVIS/JPEGImages/480p/dog --fps 24

# Batch Mode
python utils/make_video.py --input_folder data/DAVIS/JPEGImages/480p --fps 24
```

1. Track via UI:
```bash
python part2/env1/Track-Anything/launch_ui.py
```
1. Move generated masks to `results/part2_davis_eval/dog/masks/`.

2. Execute decoupled evaluation:
```bash
python part2/env1/Track-Anything/run_davis_subset.py \
    --davis_root data/DAVIS \
    --output_dir results/part2/Track-Anything/davis \
    --target_seqs dog
```

The final SOTA inpainted video frames and the tracking evaluation metrics (`track_anything_subset_metrics.json`) will be generated inside `results/part2/Track-Anything/davis/dog/`.

#### Step-by-Step Reproduction Guide (Route B: Fully Automated Global Evaluation)

To objectively measure the absolute "Tracking Upper-Bound" of the Track-Anything (XMem) model across the **entire 50-sequence DAVIS dataset** without requiring 50 manual human interventions, we developed a fully automated batch-processing script.

This script utilizes a **First-Frame GT Injection** strategy: it automatically reads the exact ground-truth mask of the very first frame for each video sequence, injects it directly into the XMem memory module as the initial prompt, and allows the model to autonomously propagate the tracking for all subsequent frames. 

**Execution Command:**
Since this bypasses the Web UI and network overhead, execution is extremely fast (approx. 25-30 FPS). Run the following command from the project root:

```bash
python part2/env1/Track-Anything/run_davis_auto.py \
    --davis_root data/DAVIS \
    --output_dir results/part2/Track-Anything/davis
```

Generated masks for all 50 sequences will be organized under `results/part2/Track-Anything/davis/[seq_name]/masks/`.

The final, overarching global tracking metrics ($\mathcal{J}_M$ and $\mathcal{J}_R$) across the entire benchmark will be aggregated and saved in `results/part2/Track-Anything/davis/track_anything_davis_global.json`.


## Part 3: Exploration - Generative Video Inpainting

In this exploration phase, we attempt to address the limitations of pure propagation-based inpainting models (like ProPainter) by introducing Generative AI (Stable Diffusion). The core idea is to generatively repair missing keyframes and use the "Injection Trick" to propagate these high-fidelity hallucinations to adjacent frames.

We designed a **Dual-Track Evaluation Pipeline**:
1. **Qualitative Track (Dynamic Mask):** Uses GT dynamic masks to evaluate visual harmony.
2. **Quantitative Track (Stationary Mask):** Uses synthetically generated random stationary masks on clean videos to force "information starvation" and evaluate structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) against Ground Truth.

### Code Explanation
- `main.py`: The orchestrator script. It provides arguments to seamlessly switch between Qualitative and Quantitative evaluation tracks, manages isolated workspaces, and extracts specific frames for direct metric comparison.
- `run_davis.py`: The batch processing evaluator. It iterates through the DAVIS dataset, injects GT masks, applies boundary dilation, and runs side-by-side upper-bound comparisons.
- `utils/diffusion_utils.py`: Contains the `run_sd_inpainting` function. It executes the Stable Diffusion pipeline and implements **Pre-Generation Dilation** (to avoid motion blur residues) and **Gaussian Feathering Blending** to soften the hard edges of generative patches.
- `utils/mask_utils.py`: Generates the random stationary masks (combinations of strokes and circles) used exclusively for the quantitative evaluation track.

### How to Run
Navigate to the project root directory before executing.

**1. Qualitative Evaluation (Visual Testing with Dynamic Masks)**
```bash
# Evaluate the Traditional Baseline (Pure ProPainter):
python part3/env1/ProPainter_Explore/main.py \
  --dataset_name bmx-trees \
  --gt_data_dir data/bmx-trees \
  --gt_mask_dir data/bmx-trees_mask \
  --method baseline

# Evaluate the Generative Approach (SD2D + ProPainter):
python part3/env1/ProPainter_Explore/main.py \
  --dataset_name bmx-trees \
  --prompt "a clean graffiti wall" \
  --gt_data_dir data/bmx-trees \
  --gt_mask_dir data/bmx-trees_mask \
  --method sd2d
```

**2. Quantitative Evaluation (Metric Testing with Stationary Masks)**
```Bash
# Evaluate the Traditional Baseline (Pure ProPainter):
python part3/env1/ProPainter_Explore/main.py \
  --dataset_name bmx-trees \
  --clean_data_dir data/bmx-trees \
  --method baseline

# Evaluate the Generative Approach (SD2D + ProPainter):
python part3/env1/ProPainter_Explore/main.py \
  --dataset_name bmx-trees \
  --prompt "a clean graffiti wall" \
  --clean_data_dir data/bmx-trees \
  --method sd2d

  python part3/env1/ProPainter_Explore/main.py \
  --dataset_name tennis \
  --prompt "a clean tennis court" \
  --clean_data_dir data/tennis \
  --method sd2d

  python part3/env1/ProPainter_Explore/main.py \
  --dataset_name my_video \
  --prompt "a clean road" \
  --clean_data_dir data/my_video \
  --method sd2d
  ```

### Outputs & Artifacts
All results for Part 3 are cleanly organized under the `results/part3_evaluation/[dataset_name]/ directory`:

`qualitative/`: Contains the .mp4 video outputs for both Baseline (Pure ProPainter) and Ours (SD + ProPainter).

`quantitative/`: Contains generated `stationary_masks/`, raw extracted video frames for visual auditing (`extracted_baseline_frames/`, `extracted_generative_frames/`), and the final metrics.json comparing SSIM and PSNR.


### Advanced Exploration: Video Diffusion with DiffuEraser (Ultimate SOTA)

To push the boundaries of our video inpainting pipeline, we integrated **DiffuEraser**, a state-of-the-art conditional video diffusion model. Unlike traditional propagation methods or 2D SD injection, DiffuEraser leverages Temporal Attention and a specialized BrushNet to deeply understand semantics and generate highly coherent backgrounds, solving complex occlusion scenarios.

Execution Command (Example on BMX-Trees):
```bash
python part3/env1/ProPainter_Explore/main.py \
  --dataset_name bmx-trees \
  --clean_data_dir data/bmx-trees \
  --method diffueraser
```


### DAVIS Dataset Evaluation
Execution Command (Batch Processing):
```bash
# Run Baseline (ProPainter) 
# For specific sequence:
python part3/env1/ProPainter_Explore/run_davis.py \
    --davis_root data/DAVIS \
    --target_seqs camel drift-chicane \
    --method baseline

# Run on the whole DAVIS dataset
python part3/env1/ProPainter_Explore/run_davis.py \
    --davis_root data/DAVIS \
    --method baseline

# Run Generative Approach (SD2D + ProPainter)
python part3/env1/ProPainter_Explore/run_davis.py \
    --davis_root data/DAVIS \
    --target_seqs camel drift-chicane \
    --method diffueraser

    # Run on the whole DAVIS dataset
python part3/env1/ProPainter_Explore/run_davis.py \
    --davis_root data/DAVIS \
    --method diffueraser
```

*(Warning: Running DiffuEraser on the entire 50-video dataset without the --target_seqs flag is extremely VRAM and time-intensive).*

#### Outputs & Artifacts:
For every evaluated sequence, the script creates a direct comparative workspace under results/part3_davis_eval/[seq_name]/:

- `gt_masks_dilated/`: The pre-processed truth masks used for the controlled experiment.

- `baseline/`: The upper-bound limit of pure temporal propagation.

- `diffueraser/`: The upper-bound limit of conditional video diffusion.