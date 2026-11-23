# Semantic Segmentation on GF-2 Satellite Imagery

Dense semantic segmentation pipeline for 16-bit raw multispectral satellite imagery from **GF-2** (*FiveBillionPixels*, 4 bands, 24 classes + unlabeled pixels mapped to `ignore_index=255`), designed to compare **FPN-MobileNetV4** and **Prithvi-EO-V2-300 (`UperNetDecoder`)** as baselines for upcoming knowledge distillation experiments.

---

## 1. Environment Setup

On the cluster, activate your configured environment (e.g., `env_tfm`), or create a local virtual environment. On FT3, it is recommended to create it in the `$STORE` directory to avoid quota issues.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate the environment when finished:

```bash
deactivate
```

## 2. Install Requirements

With the virtual environment activated, install or verify the required dependencies (including PyTorch with CUDA support, segmentation-models-pytorch, and `terratorch==0.99.7` for Prithvi):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_TFM.txt
```

## 3. Configuration (`config.json`)

The `config.json` file centralizes dataset paths, train/validation split proportions, shared training parameters, and model-specific settings:

```json
{
    "EXP_NUMBER": 1,
    "models": ["prithvi"],
    "train_dir": "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/train_16bit_raw",
    "train_gt_dir": "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/train_grayscale_masks",
    "test_dir": "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/test_16bit_raw",
    "test_gt_dir": "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/test_grayscale_masks",
    "PATCHES_DIR": "$STORE/patches_1024",
    "SAMPLES": [0.9, 0.1],
    "EPOCHS": 3,
    "BATCH_SIZE": 4,
    "PATCH_SIZE": 1024,
    "PATIENCE": 5,
    "ADA": 3,
    "OUTPUTDIR": "./results/",

    "MOBILENETV4_EPOCHS": 50,
    "MOBILENETV4_BATCH_SIZE": 4,

    "PRITHVI_EPOCHS": 3,
    "PRITHVI_BATCH_SIZE": 4,
    "PRITHVI_LR": 0.0001,
    "PRITHVI_WEIGHT_DECAY": 0.1,
    "DECODER_CHANNELS": 256,
    "BANDS": ["RED", "GREEN", "BLUE", "NIR_NARROW"],
    "NECK_INDICES": [5, 11, 17, 23]
}
```

## 4. Execution

Run the pipeline in **two stages**: first pre-extract normalized patches (`*_img.npy` and `*_mask.npy`) and per-band training percentiles (`metadata.json`) into `$STORE`, and then launch model training runs.

### Step 1: Pre-extract Patches ($STORE)

Before training, generate the pre-extracted patches required by the training dataloaders:

```bash
python preextract_patches.py --config config.json
```

Or via Slurm:

```bash
sbatch run_preextract.sh
```

### Step 2: Train Models

Once patch pre-extraction completes, run training locally or on a compute node:

```bash
python main.py --config config.json
```

You can also override the model list from the command line:

```bash
python main.py --config config.json --models mobilenetv4
```

Or submit training jobs via Slurm (for GPU training runs, e.g., NVIDIA A100):

```bash
sbatch run_job.sh
```

**Note:** Make sure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported in your launch script when training with small batch sizes and large spatial dimensions (`1024x1024`)

## 5. Outputs & Tracking

- Pre-extracted patches & percentile metadata: `$STORE/patches_1024/` (or configured `PATCHES_DIR`)
- Per-model/experiment logs: `log/results_<model>_<timestamp>.txt`
- Slurm job logs: `logs_slurm/` (depending on your `run_job.sh` configuration)
- Global metrics (JSON), checkpoints, and learning curves: `results/` (or the path specified in `OUTPUTDIR`)
- Technical notes and change logs: `docs/`
