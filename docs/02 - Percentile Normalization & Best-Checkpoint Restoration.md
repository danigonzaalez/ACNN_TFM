# 02 - Percentile Normalization & Best-Checkpoint Restoration After `training02`

## What We Detected & Why We Proposed the Change
- After the first end-to-end quarter training run completed without OOM errors, evaluation showed low metrics (`OA: 30.81%`, `AA: 4.17%`, `mIoU: 1.28%`, `fwIoU: 9.49%`), with `class_aa` collapsing onto a single dominant class and validation accuracy stalling (`0.4060`) while validation loss rose in the final epoch (`2.2215` at epoch 4 to `2.3130` at epoch 5).
- Inspecting raw 16-bit images and PNG masks revealed two key factors:
  1. Strong class imbalance across labels (`0` unlabeled and skewed frequencies among classes `1–24`, with class `3` occupying a large share of labeled pixels).
  2. Narrow effective dynamic ranges in the 4 raw bands (values roughly up to ~1000, with `p99` much lower than `65535`), meaning division by `65535.0` compressed pixel values and contrast into a tiny range near zero.
- To improve input dynamic range without data leakage, we decided to compute per-band `p1–p99` percentiles strictly on the training split and clip-scale to `[0.0, 1.0]` across train, validation, and test. We also disabled data augmentation for now to keep initial experiments clean, and updated checkpoint selection to retain and restore the weights corresponding to the best validation loss rather than the last epoch.

## Changes Made & Where
- **`utils/torch_utils.py`:**
  - Added `compute_band_percentiles(...)` to compute per-band lower and upper percentiles (`p1`, `p99`) on training images.
  - Updated `DenseQuarterDataset` to accept `p_low` and `p_high`, normalize each quarter by scaling `(x - p_low) / (p_high - p_low)` clipped to `[0.0, 1.0]` before band reordering `[2, 1, 0, 3]`.
- **`prepared_models/mobilenetv4.py`:**
  - Computed `p_low` and `p_high` exclusively from `train_images` after the train/validation split and passed them to `train_dataset`, `val_dataset`, and `test_dataset` (with `transform_flag=False`).
  - Tracked `best_val_loss` and `best_model_weights` during validation, then restored `best_model_weights` prior to testing and saving the final `.ckpt` checkpoint.