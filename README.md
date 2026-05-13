# AIAA3201 Project 3: Video Object Removal & Inpainting (Integrated Repository)

Welcome to our integrated repository for Project 3! 

This project explores the task of dynamic video object removal and background restoration, ranging from hand-crafted baseline algorithms to state-of-the-art (SOTA) video diffusion models. 

## ⚠️ Important Execution Notice: Multi-Environment Setup

To ensure the highest level of reproducibility and completely avoid dependency conflicts (especially between traditional computer vision libraries and heavy generative AI models like DiffuEraser), our team has adopted a **Decoupled Environment Strategy**.

Instead of forcing all components into a single, unstable virtual environment, we have divided the execution workspaces based on the specific architectural requirements of each pipeline. 

**DO NOT attempt to run the entire project using a single environment.** Please refer to the specific Environment Readme files (`README_ENV1.md`, `README_ENV2.md`, etc.) for exact setup and execution instructions.

## 📂 Repository Structure

Our repository is organized to physically isolate the code and dependencies for different environments while sharing the same underlying data and third-party tools.

```text
PROJECT3_INTEGRATED/
├── data/                       # Shared datasets (Tennis, BMX-Trees, DAVIS, etc.)
├── part1/                      # Baseline approach
│   ├── env1/                   # Code executable under Environment 1
│   ├── env2/                   # Code executable under Environment 2
│   └── env3/                   # Code executable under Environment 3
├── part2/                      # SOTA tracking & inpainting 
│   ├── env1/                   
│   ├── env2/                   
│   └── env3/                   
├── part3/                      # Exploration
│   ├── env1/                   
│   ├── env2/                   
│   └── env3/                   
├── third_party/                # Shared SOTA repositories (ProPainter, DiffuEraser, etc.)
├── utils/                      # Shared utility scripts (metrics, masking tools)
├── README.md                   # This master documentation
├── README_ENV1.md              # Detailed execution guide for Pipeline 1
├── requirements_env1.txt       # Dependencies for Environment 1
└── ...                         # Other environment configs (env2, env3)
```

