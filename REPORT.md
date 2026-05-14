# Report
This document summarizes the quantitative and qualitative evaluation results across all three phases of our Video Object Removal & Inpainting pipeline.

## Part1: Baseline
### 1.1 Mask R-CNN

| Dataset | Method | $\mathcal{J}_M$ (Mean IoU) | $\mathcal{J}_R$ (Recall) |
| :--- | :--- | :--- | :--- |
| **Tennis** | Mask R-CNN + Flow |0.5097| 0.8286 |
| **BMX-Trees** | Mask R-CNN + Flow | 0.1926| 0.0000 |
| **DAVIS(Global Average)** | Mask R-CNN + Flow | 0.3254| 0.3604 |

*Detailed results of DAVIS is in `\results\part1\MaskRCNN\davis`*

## Part2: Baseline
### 2.1 Track-Anything + Propainter

| Dataset | Method | $\mathcal{J}_M$ (Mean IoU) | $\mathcal{J}_R$ (Recall) |
| :--- | :--- | :--- | :--- |
| **Tennis** | Track-Anything + Propainter |0.3961| 0.5286 |
| **BMX-Trees** | Track-Anything + Propainter | 0.5160| 0.7250 |
| **DAVIS(Global Average)** | Track-Anything + Propainter | 0.9230| 0.9882 |

## Part3: Exploration
| Dataset | Method | PSNR | SSIM |
| :--- | :--- | :--- | :--- |
| **Tennis** | Propainter |25.99| 0.8422 |
| **Tennis**| Propainter + SD | 20.71|  0.7884 |
| **Tennis**| DiffuEraser |   4.72|   0.1728 |
| **BMX-Trees** | Propainter | 23.47|  0.7976 |
| **BMX-Trees** | Propainter + SD | 20.22|  0.7346 |
| **BMX-Trees** | DiffuEraser |  5.84|  0.0904 |
| **DAVIS(Global Average)** | Propainter | -| - |
 **DAVIS(Global Average)** | DiffuEraser | -| - |