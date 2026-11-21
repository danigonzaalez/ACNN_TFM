# Prithvi-EO-V2-300 Baseline & Migration to `DensePatchDataset`

## 1. Motivation & Diagnosis

During the initial run of `Prithvi-EO-V2-300` (`UperNetDecoder`) on an NVIDIA A100 GPU (~39.5 GiB VRAM) using full image quarters (`DenseQuarterDataset`, ~`3650 x 3454` pixels), a `CUDA out of memory` error occurred during the Vision Transformer forward pass. Unlike lightweight convolutional architectures such as `MobileNetV4`, Prithvi's quadratic self-attention complexity and intermediate feature activations require smaller spatial crop sizes.

Additionally, the previous per-band percentile computation (`compute_band_percentiles`) internally split each scene into 4 quarters—a logic tightly coupled to the legacy quarter-based dataset.

---

## 2. Implemented Changes

### 2.1. Patch-Based Dataset with Automatic Padding (`DensePatchDataset`)
`DensePatchDataset` was implemented in `utils/torch_utils.py` and replaced `DenseQuarterDataset` in both `prepared_models/prithvi.py` and `prepared_models/mobilenetv4.py` to ensure a fair, aligned comparison across baselines:

* **Configurable patch size:** The `"PATCH_SIZE"` key in `config.json` (currently `512`) sets the square crop dimensions `(P x P)`.
* **Automatic edge padding:** Since GF-2 scenes have dimensions `7300 x 6908` (`V x H`), bottom and right padding are computed dynamically so both axes become exact multiples of `PATCH_SIZE`:
  $$\text{pad\_v} = (P - (V \bmod P)) \bmod P, \qquad \text{pad\_h} = (P - (H \bmod P)) \bmod P$$
  * For `PATCH_SIZE = 512`: `pad_v = 380` (`7680 = 15 × 512`) and `pad_h = 260` (`7168 = 14 × 512`), yielding **210 patches per image**.
* **Border handling in images and masks:**
  * Images are padded with `0.0` *after* applying `p1–p99` percentile scaling and `[B, G, R, NIR]` band reordering, preventing artificial border zeros from skewing local pixel scaling.
  * Masks are padded with `0`, which maps to `ignore_index=255` during label remapping (`1..24 -> 0..23`, `0 -> 255`) so padded pixels do not affect the loss or evaluation metrics.

### 2.2. Full-Image Percentile Sampling (`compute_band_percentiles`)
`compute_band_percentiles` in `utils/torch_utils.py` was simplified to sample up to `max_pixels_per_image = 800_000` pixels directly across each full training image (`train_images`), preserving the exact same maximum sample budget per scene without relying on quarter splits and strictly avoiding data leakage into validation and test splits.

### 2.3. Prithvi Model Integration & Fixes (`prepared_models/prithvi.py`)
* **`UperNetDecoder` signature (`terratorch==0.99.7`):** Verified the constructor signature and updated initialization to `UperNetDecoder(embed_dim=in_channels, channels=decoder_channels)`.
* **Clean OOM exception handling:** In `PrithviUperNet.forward`, `torch.cuda.OutOfMemoryError` is explicitly re-raised before the 5D input fallback (`x.unsqueeze(2)`) to avoid redundant retries when GPU memory is exhausted.
* **Model registration (`main.py`):** Registered `"prithvi"` in `model_functions`.

### 2.4. Unified `config.json` & Documentation
* Unified `config.json` with shared parameters (`PATCH_SIZE`, `BATCH_SIZE`, `EPOCHS`, `SAMPLES`, `PATIENCE`, `ADA`, `OUTPUTDIR`) and model-prefixed hyperparameters (`PRITHVI_LR`, `PRITHVI_WEIGHT_DECAY`, `DECODER_CHANNELS`, `BANDS`, `NECK_INDICES`).
* Updated `README.md` (in English) to remove legacy *Anomaly CNNs* references and document the current GF-2 semantic segmentation workflow.

---

## 3. Current Validation Configuration

* **Target model:** `prithvi` (`Prithvi-EO-V2-300` + `UperNetDecoder`)
* **Patch resolution:** `PATCH_SIZE = 512`
* **Batch size:** `BATCH_SIZE = 2`
* **Validation epochs:** `EPOCHS = 3`
* **Optimizer:** `AdamW` (`PRITHVI_LR = 1e-4`, `PRITHVI_WEIGHT_DECAY = 0.1`) with `CosineAnnealingLR` (`ADA = 3`) and mixed precision (`AMP`).