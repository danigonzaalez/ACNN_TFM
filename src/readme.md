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
    "SAMPLES": [0.9, 0.1],
    "EPOCHS": 1,
    "BATCH_SIZE": 1,
    "PATIENCE": 5,
    "ADA": 3,
    "OUTPUTDIR": "./results/",

    "PRITHVI_LR": 0.0001,
    "PRITHVI_WEIGHT_DECAY": 0.1,
    "DECODER_CHANNELS": 256,
    "BANDS": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
    "NECK_INDICES": [5, 11, 17, 23]
}
```

## 4. Execution

### Local or Interactive Compute Node

```bash
python main.py --config config.json
```

You can also override the model list from the command line:

```bash
python main.py --config config.json --models mobilenetv4
```

### Cluster Execution via Slurm

For GPU training runs (e.g., NVIDIA A100), submit the job with `sbatch`:

```bash
sbatch run_job.sh
```

**Note:** Make sure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported in your launch script when training with small batch sizes and large spatial dimensions.

## 5. Outputs & Tracking

- Per-model/experiment logs: `log/results_<model>_<timestamp>.txt`
- Slurm job logs: `logs_slurm/` (depending on your `run_job.sh` configuration)
- Global metrics (JSON), checkpoints, and learning curves: `results/` (or the path specified in `OUTPUTDIR`)
- Technical notes and change logs: `docs/`
