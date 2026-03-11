# Forest Row Orientation Estimation

This repository provides the code, pretrained models, and evaluation datasets for the paper:

**A Learning-based Approach for Forest Row Orientation Estimation from UAV LiDAR Point Clouds (Under Review)**

The method estimates the orientation of plantation rows from UAV LiDAR point clouds using a BEV representation and a lightweight U-Net architecture (**BEV-U-Net**).

The repository includes:

- Training code
- Testing scripts for all experiments in the paper
- Pretrained models
- Evaluation datasets

------

# Repository Structure

```
repo/
│
├── train.py
├── test1.py
├── test2.py
├── test3.py
│
├── model/
│   ├── 1_model_main.pth
│   ├── 2_model_with_forest_yaw.pth
│   ├── 3_model_yaw_0-120.pth
│   ├── 4_ablation_plain.pth
│   ├── 5_ablation_wo_sign-inv.pth
│   └── 6_ablation_wo_tv.pth
│
└── val_data/
    ├── val/
    │   ├── pcd/
    │   ├── json/
    │   ├── pcd_cutout_10_10/
    │   ├── pcd_cutout_20_20/
    │   └── pcd_cutout_30_30/
    │
    ├── val_0_120/
    │   ├── pcd/
    │   └── json/
    │
    ├── val_0_180/
    │   ├── pcd/
    │   └── json/
    │
    └── val_120_180/
        ├── pcd/
        └── json/
```

------

# Model Files

The `model` folder contains pretrained models used in different experiments.

| Model file                      | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **1_model_main.pth**            | BEV-U-Net model used in the main experiments (Section 5.2, Section 5.5, and Appendix A). |
| **2_model_with_forest_yaw.pth** | Model trained with randomized plantation orientations for the experiment in **Section 5.4.1 (Randomized Plantation Orientation)**. |
| **3_model_yaw_0-120.pth**       | Model trained for the **orientation generalization experiment** in **Section 5.4.2**. |
| **4_ablation_plain.pth**        | Plain baseline model used in the **ablation study (Appendix A)**. |
| **5_ablation_wo_sign-inv.pth**  | Model trained **without the sign-invariant cosine loss** (Appendix A). |
| **6_ablation_wo_tv.pth**        | Model trained **without the TV regularization** (Appendix A). |

------

# Validation Datasets

The `val_data` folder contains the evaluation datasets used in the experiments.

### 1. Main validation dataset

```
val_data/val/
```

Contains the original validation point clouds and their metadata.

```
val/
├── pcd/
├── json/
```

------

### 2. Occlusion experiments (Section 5.5)

To evaluate robustness under occlusion, artificial cutouts are applied to the point clouds.

```
val/
├── pcd_cutout_10_10
├── pcd_cutout_20_20
└── pcd_cutout_30_30
```

These correspond to the occlusion scenarios reported in **Section 5.5**.

------

### 3. Randomized plantation orientation dataset

```
val_data/val_0_180/
```

Used in **Section 5.4.1 (Randomized Plantation Orientation)**.

Forest orientation range:

```
(-90°, 90°)
```

------

### 4. Orientation generalization datasets

Used in **Section 5.4.2 (Unseen Orientation Generalization)**.

Seen orientations:

```
val_data/val_0_120
forest orientation range: (-30°, 90°)
```

Unseen orientations:

```
val_data/val_120_180
forest orientation range: (-90°, -30°)
```

------

# Training

Training can be performed using:

```
python train.py
```

The training script implements the BEV-U-Net model and the loss functions described in the paper.

------

# Testing

Three testing scripts are provided for different experiments.

------

## test1.py

Used for experiments in:

```
Section 5.2  Main results
Section 5.5  Occlusion experiments
Appendix A  blation Study
```

To switch experiments, modify:

```
pcd_dir
json_dir
ckpt_path
```

Example configurations:

### Section 5.2 (original dataset)

```
pcd_dir = "val_data/val/pcd"
json_dir = "val_data/val/json"
```

### Section 5.5 (10×10 occlusion)

```
pcd_dir = "val_data/val/pcd_cutout_10_10"
json_dir = "val_data/val/json"
```

### Section 5.5 (20×20 occlusion)

```
pcd_dir = "val_data/val/pcd_cutout_20_20"
json_dir = "val_data/val/json"
```

### Section 5.5 (30×30 occlusion)

```
pcd_dir = "val_data/val/pcd_cutout_30_30"
json_dir = "val_data/val/json"
```

------

## test2.py

Used for experiments in:

```
Section 5.4.1 Randomized Plantation Orientation
Section 5.4.2 Unseen Orientation Generalization
```

To switch experiments, modify:

```
pcd_dir
json_dir
ckpt_path
```

Example configurations:

### Section 5.4.1 Randomized orientation

```
pcd_dir = "val_data/val_0_180/pcd"
json_dir = "val_data/val_0_180/json"
ckpt_path = "model/2_model_with_forest_yaw.pth"
```

### Section 5.4.2 Seen orientations

```
pcd_dir = "val_data/val_0_120/pcd"
json_dir = "val_data/val_0_120/json"
ckpt_path = "model/3_model_yaw_0-120.pth"
```

### Section 5.4.2 Unseen orientations

```
pcd_dir = "val_data/val_120_180/pcd"
json_dir = "val_data/val_120_180/json"
ckpt_path = "model/3_model_yaw_0-120.pth"
```

------

## test3.py

Used for the **ablation study in Appendix A**.

To run different ablation settings, modify:

```
pcd_dir
json_dir
ckpt_path
```

Example configurations:

Plain baseline:

```
ckpt_path = "model/4_ablation_plain.pth"
```

Without sign-invariant loss:

```
ckpt_path = "model/5_ablation_wo_sign-inv.pth"
```

Without TV regularization:

```
ckpt_path = "model/6_ablation_wo_tv.pth"
```

The results for the **Full model** and **w/o histogram aggregation** can be obtained using `test1.py`.

Note:

- **Pixel MAE / Pixel Acc** use histogram aggregation.
- **Frame Err / Frame Acc** are computed without histogram aggregation.

The loss function used during training does not affect the testing results.



