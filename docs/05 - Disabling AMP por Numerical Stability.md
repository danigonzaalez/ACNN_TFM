# Disabling Automatic Mixed Precision (AMP) for Numerical Stability

## 1. Observed Issue (`NaN` Loss Divergence)

During multi-epoch training of `FPN-MobileNetV4` (`PATCH_SIZE = 1024`, `BATCH_SIZE = 4`, `lr = 0.001`), the training and validation losses diverged to `nan` starting at Epoch 3, causing validation accuracy to collapse to ~`1.31%`:

```text
Epoch:   1/20 | Train Loss: 2.2848 | Val Loss: 2.2930 | Acc: 0.3223
Epoch:   2/20 | Train Loss: 1.9374 | Val Loss: 2.6069 | Acc: 0.2466
Epoch:   3/20 | Train Loss: nan    | Val Loss: nan    | Acc: 0.3407
Epoch:   4/20 | Train Loss: nan    | Val Loss: nan    | Acc: 0.0131
```

## 2. Root Cause

* Limited dynamic range of float16 (AMP + GradScaler): By default, torch.amp.autocast("cuda") executes eligible operations in float16. With larger 1024 x 1024 spatial patches and AdamW, large intermediate activations or gradient spikes caused float16 numerical overflow (Inf/NaN), permanently corrupting model weights and turning subsequent loss averages into nan.

* Original rationale no longer required: Mixed precision (AMP) had originally been introduced to fit large quarter-scene crops (~3650 x 3454) into GPU VRAM. After migrating to pre-extracted 1024 x 1024 patches with configurable per-model batch sizes (MOBILENETV4_BATCH_SIZE and PRITHVI_BATCH_SIZE), full float32 execution fits within the NVIDIA A100's 40 GB VRAM without needing float16 memory savings.

## 3. Solution Implemented

* Removed torch.amp.autocast("cuda") and torch.amp.GradScaler from the training, validation, and testing loops in both src/prepared_models/mobilenetv4.py and src/prepared_models/prithvi.py.

* Switched both models to full float32 precision (outputs = model(inputs), loss.backward(), optimizer.step()), ensuring stable gradient updates and fair, consistent numerical precision across experiments.