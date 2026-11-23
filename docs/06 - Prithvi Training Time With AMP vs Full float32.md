# Prithvi Training Time: With AMP vs. Full `float32`

Disabling AMP (`float16` + `GradScaler`) was initially adopted because VRAM constraints from large quarter-scene crops were no longer an issue with `1024 x 1024` patches, and because `float16` mixed precision had previously caused gradient/overflow `NaN` losses during `MobileNetV4` training.

However, removing AMP in `Prithvi-EO-V2-300` (`1024 x 1024`, `batch_size = 4`, full `float32`) increased epoch runtime from **~14.5 minutes** (with AMP) to **~1h 26m per epoch** (~`3.31 s/it` in training), making multi-epoch runs much tighter against cluster wall-clock limits.

For longer runs on the A100, it is worth considering either re-enabling standard AMP or using `bfloat16` (which offers wider dynamic range than `float16` to reduce `NaN` risk without needing `GradScaler`) while preserving faster training times.