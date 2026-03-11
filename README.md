# Forest Row Orientation Estimation

This repository provides the code, pretrained models, and evaluation datasets for the paper:

**A Learning-based Approach for Forest Row Orientation Estimation from UAV LiDAR Point Clouds (Under Review)**

The method estimates plantation row orientation from UAV LiDAR point clouds using a BEV representation and a U-Net architecture (**BEV-U-Net**).

The repository includes:

- training code (`train.py`)
- testing scripts for the experiments in the paper (`test1.py`, `test2.py`, `test3.py`)
- pretrained models (`model/`)
- validation datasets (`val_data/`)

------

# Model Files

The `model` folder contains pretrained models used in different experiments.

- **1_model_main.pth**
  BEV-U-Net model used in the main experiments (Section 5.2, Section 5.5 and Appendix A).
- **2_model_with_forest_yaw.pth**
  Model trained with randomized plantation orientations for the experiment in **Section 5.4.1 (Randomized Plantation Orientation)**.
- **3_model_yaw_0-120.pth**
  Model used for the **orientation generalization experiment** in **Section 5.4.2**.
- **4_ablation_plain.pth**
  Plain baseline model used in the **ablation study ** (Appendix A).
- **5_ablation_wo_sign-inv.pth**
  Model trained **without the sign-invariant cosine loss** (Appendix A).
- **6_ablation_wo_tv.pth**
  Model trained **without the TV regularization** (Appendix A).

------

# Validation Data

The `val_data` folder contains the validation datasets used in the experiments.

### Main validation dataset

```
val_data/val
```

Contains the original validation point clouds and metadata.

This folder also includes occlusion variants used in **Section 5.5**:

- `pcd_cutout_10_10`
- `pcd_cutout_20_20`
- `pcd_cutout_30_30`

------

### Randomized plantation orientation dataset

```
val_data/val_0_180
```

Used in **Section 5.4.1 (Randomized Plantation Orientation)**.

------

### Orientation generalization datasets

```
val_data/val_0_120
val_data/val_120_180
```

Used in **Section 5.4.2 (Unseen Orientation Generalization)**.

------

# Training

Training can be performed using

```
python train.py
```

------

# Testing

Three testing scripts are provided:

- **test1.py**
  Used for experiments in **Section 5.2 (main results)，Section 5.5 (occlusion experiments) and ablation study in Appendix A**.
- **test2.py**
  Used for experiments in **Section 5.4.1 (Randomized Plantation Orientation)** and
  **Section 5.4.2 (Unseen Orientation Generalization)**.
- **test3.py**
  Used for the **ablation study in Appendix A**.

For each script, only the following paths need to be modified:

```
pcd_dir
json_dir
ckpt_path
```

------

# Notes

- **Pixel MAE / Pixel Acc** correspond to results using histogram aggregation.
- **Frame Err / Frame Acc** correspond to results without histogram aggregation.
- The loss function used during training does **not affect the testing results**, since evaluation relies only on model predictions.
