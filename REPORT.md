# Report
This document summarizes the quantitative and qualitative evaluation results across all three phases of our Video Object Removal & Inpainting pipeline.

## Part1: Baseline

| Dataset | Method | $\mathcal{J}_M$ (Mean IoU) | $\mathcal{J}_R$ (Recall) |
| :--- | :--- | :--- | :--- |
| **Tennis** | Mask R-CNN + Flow |0.5097| 0.8286 |
| **BMX-Trees** | Mask R-CNN + Flow | 0.1926| 0.0000 |
| **DAVIS(Global Average)** | Mask R-CNN + Flow | 0.3254| 0.3604 |
| **Tennis** | YOLO |0.2724| 0.1125 |
| **BMX-Trees** | YOLO | 0.5954| 0.7692 |
| **DAVIS(Global Average)** | YOLO | 0.327| 0.309 |

*Detailed results of DAVIS is in `\results\part1\MaskRCNN\davis`*

## Part2: Baseline

| Dataset | Method | $\mathcal{J}_M$ (Mean IoU) | $\mathcal{J}_R$ (Recall) |
| :--- | :--- | :--- | :--- |
| **Tennis** | Track-Anything + Propainter |0.3961| 0.5286 |
| **BMX-Trees** | Track-Anything + Propainter | 0.5160| 0.7250 |
| **DAVIS(Global Average)** | Track-Anything + Propainter | 0.9230| 0.9882 |
| **Tennis** | SAM2 + Propainter |0.6488| 0.5713 |
| **BMX-Trees** | SAM2 + Propainter | 0.8857| 0.9713 |
| **DAVIS(Global Average)** | SAM2 + Propainter  | 0.9180     | 0.9898|
| **Tennis** | Cutie + Propainter |0.6433| 1.000 |
| **BMX-Trees** | Cutie + Propainter |0.6433| 1.0000 |
| **Tennis** | VGGT4D + Propainter |0.5211| 0.5713 |
| **BMX-Trees** | VGGT4D + Propainter | 0.8857| 0.9713 |
| **DAVIS(Global Average)** | VGGT4D + Propainter | 0.563| 0.628 |


note: Track-Anything on DAVIS dataset and Cutie on tennis & bmx-trees datasets are using the ground truth mask in the first frame.

## Part3: Exploration
| Dataset | Method | PSNR | SSIM |
| :--- | :--- | :--- | :--- |
| **Tennis** | Propainter |25.99| 0.8422 |
| **Tennis**| Propainter + SD | 20.71|  0.7884 |
| **Tennis**| DiffuEraser |   19.31|   0.5741 |
| **BMX-Trees** | Propainter | 23.47|  0.7976 |
| **BMX-Trees** | Propainter + SD | 20.22|  0.7346 |
| **BMX-Trees** | DiffuEraser |  17.01|  0.5658 |
| **DAVIS(Global Average)** | Propainter | 26.86| 0.8614 |
 **DAVIS(Global Average)** | DiffuEraser | -| - |


 | Dataset | Method | $\mathcal{J}_M$ (Mean IoU) | $\mathcal{J}_R$ (Recall) |
| :--- | :--- | :--- | :--- |
| **Tennis** |VGGT4D +pi3 (in 10 frames) |0.0550| 0.0000 |
| **BMX-Trees** | VGGT4D +pi3 (in 30 frames) | 0.1325| 0.0000 |
| **Tennis** |VGGT4D +pi3 (Hybrid) (in 10 frames) |0.5094| 0.7000 |
| **BMX-Trees** | VGGT4D +pi3 (Hybrid) (in 30 frames)| 0.5282| 0.8000 |
| **Tennis** |sam3 + VGGT4D |0.4301| 0.8571 |
| **BMX-Trees** | sam3 + VGGT4D | 0.3544| 0.4250 |
| **Tennis** |VGGT4D + map-anything |05088| 0.7000 |
| **BMX-Trees** | VGGT4D + map-anything | 0.5480| 1.0000 |
| **Tennis** |VGGT4D + map-anything + SAM2 |0.5406| 1.0000 |
| **BMX-Trees** | VGGT4D + map-anything + SAM2 | 0.5445| 1.0000 |