# Diff-Instruct-star
We introduce DI\*-SDX-1step Model, which is a leading human-preferred 1-step text-to-image model of 1024 resolution.


![Figure from PDF](assets/big_demo_pr1.pdf "Figure from PDF")


Below are the tables from the paper converted into Markdown format:

### Table 1: Quantitative comparisons of 1024 × 1024 resolution leading text-to-image models

| Model                        | Steps ↓ | Type   | Params ↓ | Image Reward ↑ | AES Score ↑ | PickScore ↑ | CLIPScore ↑ | Inference Time ↓ (per 10 images) |
|------------------------------|---------|--------|----------|----------------|-------------|-------------|-------------|--------------------------------------|
|||||||||**Multi-step Models**|
| SDXL-BASE (Podell et al., 2023)      | 50      | UNET   | 2.6B    | 0.887          | 5.72        | 0.2274      | 32.72      | 111 sec                          |
| SDXL-DPO (Wallace et al., 2024)     | 50      | UNET   | 2.6B    | 1.102          | 5.77        | 0.2290      | **33.03**      | 111 sec                          |
| SD3.5-LARGE (SD3)            | 28      | DIT    | 8B       | **1.133**          | 5.70        | 0.2306      | 32.70      | 66.23 sec                        |
| FLUX-DEV (FLU)               | 50      | DIT    | 12B      | 1.132          | **5.90**        | **0.2317**      | 31.70      | 118.64 sec                       |
|||||||||**1-step Models**|
| DMD2-SDXL (Yin et al., 2024) | 1       | UNET   | 2.6B    | 0.930          | 5.51        | 0.2249      | 32.97      | 2.22 sec                         |
| DIFF-INSTRUCT (Luo et al., 2024b)  | 1       | UNET   | 2.6B    | 1.058          | 5.60        | 0.2253      | **33.02**      | 2.22 sec                         |
| SIM (Luo et al., 2024c)     | 1       | UNET   | 2.6B    | 1.049          | 5.66        | 0.2273      | 32.93      | 2.22 sec                         |
| DIFF-INSTRUCT++-SDXL (Luo, 2024)  | 1       | UNET   | 2.6B    | 1.061          | 5.58        | 0.2260      | 32.94      | 2.22 sec                         |
| **DI\*-SDXL (Ours)**          | 1       | UNET   | 2.6B    | 1.067          | 5.74        | 0.2304      | 32.82      | 2.22 sec                         |
| **DI\*-SDXL (Longer Training)** | **1**       | UNET   | **2.6B**    | **1.140**          | **5.83**        | **0.2331**      | 32.75      | **2.22 sec**                         |

---

### Table 2: Ablation study on Parti Prompts of DI*-SDXL-1step models

| Model                          | Steps | Params | Image Reward ↑ | AES Score ↑ | PickScore ↑ | CLIPScore ↑ | ($\alpha_{rev}$, \alpha_{cfg}) |
|--------------------------------|-------|--------|----------------|-------------|-------------|-------------|--------------|
| DMD2-SDXL (Init Model)         | 1     | 2.6B   | 0.938          | 5.51        | 0.2249      | 32.97       | -            |
| DI++-SDXL (Aligned Using KL)  | 1     | 2.6B   | 0.846          | 5.50        | 0.2243      | 32.66       | (0, 0)       |
| DI++-SDXL (Equ to Diff-Instruct) | 1     | 2.6B   | 1.058          | 5.60        | 0.2253      | 33.02       | (0, 7.5)     |
| DI++-SDXL (Aligned Using KL)  | 1     | 2.6B   | 1.061          | 5.58        | 0.2260      | 32.94       | (100, 7.5)   |
| DI\*-OUT-SDXL (Out CFG)         | 1     | 2.6B   | 1.082          | 5.63        | 0.2263      | **33.03**       | (100, 7.5)   |
| DI\*-IN-SDXL (Baseline, No Reward) | 1     | 2.6B   | 0.782          | 5.74        | 0.2256      | 32.16       | (0, 0)       |
| DI\*-IN-SDXL (Equ to SIM, Only CFG) | 1     | 2.6B   | 1.049          | 5.66        | 0.2273      | 32.93       | (0, 7.5)     |
| DI\*-IN-SDXL (Human Reward + CFG) | 1     | 2.6B   | 1.031          | 5.69        | 0.2274      | 32.87       | (1, 7.5)     |
| DI\*-IN-SDXL                   | 1     | 2.6B   | 1.048          | 5.66        | 0.2278      | 32.91       | (10, 7.5)    |
| DI\*-IN-SDXL                   | 1     | 2.6B   | 1.020          | 5.68        | 0.2278      | 32.82       | (100, 4.5)   |
| **DI\*-IN-SDXL**                | 1     | 2.6B   | 1.067          | 5.74        | 0.2304      | 32.82       | (100, 7.5)   |
| **DI\*-IN-SDXL (Longer Training)** | **1**     | **2.6B**   | **1.140**          | **5.83**        | **0.2331**      | 32.75       | (100, 7.5)   |

---

### Table 3: Quantitative evaluations on HPSv2.1 scores

| Model                          | Animation ↑ | Concept Art ↑ | Painting ↑ | Photo ↑ | Average ↑ |
|--------------------------------|-------------|---------------|------------|---------|-----------|
| 50STEP-SDXL-BASE (Podell et al., 2023) | 30.85       | 29.30         | 28.98      | 27.05   | 29.05     |
| 50STEP-SDXL-DPO (Wallace et al., 2024) | 32.01       | 30.75         | 30.70      | 28.24   | 30.42     |
| 28STEP-SD3.5-LARGE             | 31.89       | 30.19         | 30.39      | 28.01   | 30.12     |
| 50STEP-FLUX-DEV                | 32.09       | 30.44         | 31.17      | **29.09**   | 30.70     |
| 1STEP-DMD2-SDXL (Yin et al., 2024) | 29.72       | 27.96         | 27.64      | 26.55   | 27.97     |
| 1STEP-DIFF-INSTRUCT-SDXL (Luo et al., 2024b) | 31.15       | 29.71         | 29.72      | 28.20   | 29.70     |
| 1STEP-SIM-SDXL (Luo et al., 2024c) | 31.97       | 30.46         | 30.13      | 28.08   | 30.16     |
| 1STEP-DI++-SDXL (Luo, 2024)    | 31.19       | 29.88         | 29.61      | 28.21   | 29.72     |
| **1STEP-DI\*-SDXL (Ours)**      | 32.26       | 30.57         | 30.10      | 27.95   | 30.22     |
| **1STEP-DI\*-SDXL (Ours, Longer Training)** | **33.22**       | **31.67**         | **31.25**      | 28.62   | **31.19**     |
