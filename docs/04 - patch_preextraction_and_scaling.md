# Patch Pre-Extraction to `$STORE` & Training Throughput Optimization

## 1. Motivation & Bottleneck Diagnosis

During the initial patch-based experiments (`PATCH_SIZE = 512`, `BATCH_SIZE = 2`), both `MobileNetV4` (`1.78 s/it`) and `Prithvi-EO-V2-300` (`1.79 s/it`) exhibited virtually identical iteration speeds and hit the 6-hour Slurm wall-time limit after completing only a single training epoch (`11,340` iterations).

Because a lightweight convolutional network (`MobileNetV4`) and a 300M-parameter Vision Transformer (`Prithvi`) ran at the exact same pace, profiling confirmed that **the primary bottleneck was disk I/O and CPU preprocessing rather than GPU compute**:
* In `DensePatchDataset.__getitem__`, every single patch request re-read the entire `7300 x 6908 x 4` `.raw` scene and `.png` mask from Lustre, cast to `float32`, computed percentile scaling, reordered bands, padded the scene, and discarded >99% of the image after cropping a single patch (`210` redundant full-scene reads per image per epoch).
* Additionally, small spatial crops (`512 x 512`) combined with a tiny batch size (`BATCH_SIZE = 2`) generated `11,340` training iterations per epoch, underutilizing the parallel compute capacity of the NVIDIA A100 GPU.

---

## 2. Implemented Changes

### 2.1. Offline Patch Pre-Extraction (`src/preextract_patches.py` & `run_preextract.sh`)
To eliminate redundant full-scene I/O during training and across model comparisons, an offline preprocessing pipeline was created:
* **Leakage-free percentile computation:** Reproduces the exact deterministic `train`/`val` split (`SAMPLES`, `random.seed(42)`) and computes per-band `p1` and `p99` percentiles strictly over `train_images`.
* **Normalized & padded patch export:** Applies `p1–p99` scaling clipped to `[0.0, 1.0]`, reorders channels to `[R, G, B, NIR]`, pads bottom/right borders to exact multiples of `PATCH_SIZE`, remaps labels (`1..24 -> 0..23`, `0 -> 255`), and saves each patch to `$STORE/patches_<PATCH_SIZE>/{train,val,test}`:
  * Image patches (`*_img.npy`): stored in channel-first layout `(C, P, P)` as `float16` to reduce storage footprint and disk read latency.
  * Mask patches (`*_mask.npy`): stored as `uint8` `(P, P)`.
  * Metadata (`metadata.json`): records scene counts, `patch_size`, and the computed `p_low` / `p_high` arrays for reproducibility.
* **Storage footprint within FT3 `$STORE` quotas (`500 GB` / `300,000` files):** At `PATCH_SIZE = 1024`, each `7300 x 6908` scene pads to `8192 x 7168` (`8 x 7 = 56` patches/image). Across all `150` scenes, this produces `8,400` image patches and `8,400` mask patches (`16,800` files total, ~`75–80 GB`), comfortably within `$STORE` limits.

### 2.2. Fast Pre-Extracted Dataset (`PreextractedPatchDataset` in `src/utils/torch_utils.py`)
Implemented `PreextractedPatchDataset` to load pre-normalized `*_img.npy` and `*_mask.npy` pairs directly from `$STORE`, converting them to `(float32, int64)` tensors and optionally applying synchronized random horizontal/vertical flips.

### 2.3. Scaling Patch Size, Batch Size & Model-Specific Epochs
* **Higher spatial resolution & batch parallelism:** Updated `config.json` to `PATCH_SIZE = 1024` and `BATCH_SIZE = 8` (alongside `PATCHES_DIR = "$STORE/patches_1024"`), reducing training iterations per epoch from `11,340` down to `756` (`108 * 56 / 8`).
* **Model-specific epoch budgets:** Added `MOBILENETV4_EPOCHS` (`5`) and `PRITHVI_EPOCHS` (`3`) in `config.json`, `src/prepared_models/mobilenetv4.py`, and `src/prepared_models/prithvi.py` so both models can be launched independently against the shared `config.json` via `--models <model_name>` without file collisions.
* **Band order alignment in Prithvi:** Updated `BANDS` default and config to `["RED", "GREEN", "BLUE", "NIR_NARROW"]` to match the `[R, G, B, NIR]` channel order produced by the dataset pipeline.