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
