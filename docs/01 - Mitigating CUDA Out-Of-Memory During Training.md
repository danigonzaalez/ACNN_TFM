# 01 - Mitigating CUDA Out-Of-Memory During Training

## What We Detected & Why We Proposed the Change
- Training `FPN-MobileNetV4` (`tu-mobilenetv4_conv_medium`) on full image quarters (`3680 x 3456` after padding to multiples of 32) on a ~39.5 GiB A100 GPU triggered a `torch.OutOfMemoryError` during the `backward()` pass when `BATCH_SIZE` was set to `2`.
- Storing intermediate activations across large spatial quarters for gradient computation exceeded available VRAM, and memory fragmentation added overhead.
- To fit training within VRAM while preserving stability and throughput on the A100, we proposed reducing the batch size to `1`, enabling Automatic Mixed Precision (`torch.amp.autocast` with `GradScaler`) to cut activation memory usage and speed up compute, and configuring expandable CUDA memory segments.

## Changes Made & Where
- **`config.json`:** Set `"BATCH_SIZE": 1` (and updated the fallback default in `prepared_models/mobilenetv4.py` to `1`).
- **`run_job.sh`:** Exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prior to launching `main.py` to reduce CUDA memory fragmentation.
- **`prepared_models/mobilenetv4.py`:**
  - Initialized `scaler = torch.amp.GradScaler("cuda", enabled=cuda)`.
  - Wrapped forward pass and loss computation in `torch.amp.autocast("cuda", enabled=cuda)` during training, validation, and testing, and scaled gradients with `scaler` during training.