import os
import copy
import glob
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models.decoders import UperNetDecoder

from utils.torch_utils import DensePatchDataset, EarlyStopping, compute_band_percentiles
from utils.metrics import SegmentationMetrics
from utils.general_utils import plot_learning_curves


class PrithviUperNet(nn.Module):
    """
    Wrapper around a pretrained Prithvi backbone, feature-map neck,
    UperNetDecoder, and segmentation head outputting (B, num_classes, H, W).
    """

    def __init__(
        self,
        backbone_name="timm_prithvi_eo_v2_300",
        bands=None,
        pretrained=True,
        neck_indices=None,
        decoder_channels=256,
        num_classes=24,
        head_dropout=0.1,
    ):
        super().__init__()
        if bands is None:
            bands = ["BLUE", "GREEN", "RED", "NIR_NARROW"]
        if neck_indices is None:
            neck_indices = [5, 11, 17, 23]

        self.neck_indices = neck_indices

        # Build pretrained Prithvi backbone with explicit band ordering
        build_kwargs = {
            "pretrained": pretrained,
            "bands": bands,
        }
        try:
            self.backbone = BACKBONE_REGISTRY.build(backbone_name, **build_kwargs)
        except TypeError:
            # Fallback in case a specific registry entry does not accept `bands`
            self.backbone = BACKBONE_REGISTRY.build(backbone_name, pretrained=pretrained)

        # Determine feature channels for the selected transformer blocks
        if hasattr(self.backbone, "out_channels"):
            all_channels = list(self.backbone.out_channels)
            in_channels = [all_channels[i] for i in neck_indices]
        elif hasattr(self.backbone, "embed_dim"):
            in_channels = [self.backbone.embed_dim] * len(neck_indices)
        else:
            raise AttributeError("Could not infer feature channel dimensions from Prithvi backbone.")

        # ReshapeTokensToImage neck (built cleanly in PyTorch for spatial token grids)
        self.decoder = UperNetDecoder(
            embed_dim=in_channels,
            channels=decoder_channels,
        )
        self.dropout = nn.Dropout2d(p=head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def _tokens_to_image(self, feat, h_in, w_in):
        """
        Converts ViT output features into 2D spatial feature maps (B, C, H', W').
        Handles both already-2D tensors and (B, N_tokens, C) / (B, 1 + N_tokens, C) token sequences.
        """
        if feat.ndim == 4:
            return feat

        if feat.ndim != 3:
            raise ValueError(f"Unexpected feature tensor shape from backbone: {feat.shape}")

        b, seq_len, c = feat.shape
        patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            ph, pw = patch_size[0], patch_size[1]
        else:
            ph = pw = int(patch_size)

        grid_h, grid_w = h_in // ph, w_in // pw
        num_spatial_tokens = grid_h * grid_w

        # Drop CLS or extra tokens if present at the beginning of the sequence
        if seq_len > num_spatial_tokens:
            feat = feat[:, seq_len - num_spatial_tokens :, :]

        return feat.transpose(1, 2).reshape(b, c, grid_h, grid_w).contiguous()

    def forward(self, x):
        _, _, h_in, w_in = x.shape

        # Backbone forward (some Prithvi backbones accept 4D (B, C, H, W) or 5D (B, C, T, H, W))
        try:
            backbone_out = self.backbone(x)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            backbone_out = self.backbone(x.unsqueeze(2))

        if isinstance(backbone_out, dict):
            feats = list(backbone_out.values())
        elif isinstance(backbone_out, (list, tuple)):
            feats = list(backbone_out)
        else:
            feats = [backbone_out]

        # SelectIndices equivalent
        if len(feats) > max(self.neck_indices):
            selected_feats = [feats[i] for i in self.neck_indices]
        else:
            selected_feats = feats

        # ReshapeTokensToImage equivalent
        spatial_feats = [self._tokens_to_image(f, h_in, w_in) for f in selected_feats]

        decoded = self.decoder(spatial_feats)
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[-1]

        logits = self.classifier(self.dropout(decoded))

        # Ensure output matches input spatial resolution (H_in, W_in)
        if logits.shape[-2:] != (h_in, w_in):
            logits = F.interpolate(logits, size=(h_in, w_in), mode="bilinear", align_corners=False)

        return logits


def main(EXP_NUMBER, config):
    print(f"* Prithvi exp: {EXP_NUMBER}")

    # Directory and parameter configuration from config.json
    TRAIN_DIR = config.get("train_dir")
    TRAIN_GT_DIR = config.get("train_gt_dir")
    TEST_DIR = config.get("test_dir")
    TEST_GT_DIR = config.get("test_gt_dir")

    SAMPLES = config.get("SAMPLES", [0.9, 0.1])
    default_epochs = config.get("EPOCHS", 50)
    EPOCHS = config.get("PRITHVI_EPOCHS", default_epochs)
    BATCH_SIZE = config.get("BATCH_SIZE", 2)
    PATCH_SIZE = config.get("PATCH_SIZE", 512)
    PATIENCE = config.get("PATIENCE", 5)
    ADA = config.get("ADA", 3)
    LR = config.get("PRITHVI_LR", 1e-4)
    WEIGHT_DECAY = config.get("PRITHVI_WEIGHT_DECAY", 0.1)
    DECODER_CHANNELS = config.get("DECODER_CHANNELS", 256)
    OUTPUTDIR = config.get("OUTPUTDIR", "./results/")
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # Bands aligned to DenseQuarterDataset output order: [B, G, R, NIR]
    BANDS = config.get("BANDS", ["BLUE", "GREEN", "RED", "NIR_NARROW"])
    NECK_INDICES = config.get("NECK_INDICES", [5, 11, 17, 23])

    print(f"* Train Directory: {TRAIN_DIR}")
    print(f"* Train GT Directory: {TRAIN_GT_DIR}")
    print(f"* Test Directory: {TEST_DIR}")
    print(f"* Test GT Directory: {TEST_GT_DIR}")
    print(f"* Samples proportions: {SAMPLES}")
    print(f"* Epochs: {EPOCHS}")
    print(f"* Batch Size: {BATCH_SIZE}")
    print(f"* Patch Size: {PATCH_SIZE}")
    print(f"* Learning Rate: {LR} | Weight Decay: {WEIGHT_DECAY}")
    print(f"* Early Stopping Patience: {PATIENCE}")
    print(f"* ADA (LR Scheduler): {ADA}")
    print(f"* Bands: {BANDS} | Neck Indices: {NECK_INDICES}")
    print(f"* Output Directory: {OUTPUTDIR}")

    time_start = time.time()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    print(f"* Device: {device}")

    all_train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.raw")))
    all_train_masks = sorted(glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.raw")))
    test_masks = sorted(glob.glob(os.path.join(TEST_GT_DIR, "*.png")))

    if len(all_train_images) == 0 or len(test_images) == 0:
        raise ValueError("No .raw images found. Please check config.json directories.")

    H, V, nclases = 6908, 7300, 24

    # Same deterministic split as MobileNetV4
    combined_train = list(zip(all_train_images, all_train_masks))
    random.seed(42)
    random.shuffle(combined_train)
    all_train_images, all_train_masks = zip(*combined_train)

    total_proportion = SAMPLES[0] + SAMPLES[1]
    val_ratio = SAMPLES[1] / total_proportion if total_proportion > 0 else 0
    num_val_images = int(len(all_train_images) * val_ratio)

    if num_val_images > 0:
        val_images = list(all_train_images[:num_val_images])
        val_masks = list(all_train_masks[:num_val_images])
        train_images = list(all_train_images[num_val_images:])
        train_masks = list(all_train_masks[num_val_images:])
    else:
        val_images, val_masks = [], []
        train_images = list(all_train_images)
        train_masks = list(all_train_masks)

    print(
        f"* Split -> Train images: {len(train_images)} | "
        f"Val images: {len(val_images)} | Test images: {len(test_images)}"
    )

    # Compute per-band percentiles (p1, p99) strictly on train images to avoid leakage
    p_low, p_high = compute_band_percentiles(train_images, H=H, V=V, B=4, q_low=1.0, q_high=99.0)
    print(f"* Train percentiles p1: {p_low.tolist()} | p99: {p_high.tolist()}")

    # Train, validation, and test datasets and loaders
    train_dataset = DensePatchDataset(train_images, train_masks, H, V, patch_size=PATCH_SIZE, transform_flag=False, p_low=p_low, p_high=p_high)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    if len(val_images) > 0:
        val_dataset = DensePatchDataset(val_images, val_masks, H, V, patch_size=PATCH_SIZE, transform_flag=False, p_low=p_low, p_high=p_high)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    test_dataset = DensePatchDataset(test_images, test_masks, H, V, patch_size=PATCH_SIZE, transform_flag=False, p_low=p_low, p_high=p_high)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = PrithviUperNet(
        backbone_name="timm_prithvi_eo_v2_300",
        bands=BANDS,
        pretrained=True,
        neck_indices=NECK_INDICES,
        decoder_channels=DECODER_CHANNELS,
        num_classes=nclases,
        head_dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if ADA == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[EPOCHS // 2, (5 * EPOCHS) // 6], gamma=0.1
        )
    elif ADA == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
    elif ADA == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    print(f"* Training Prithvi-EO-V2-300 (UperNetDecoder), exp.{EXP_NUMBER}")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        if ADA > 1 and scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=cuda):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    valid_mask = (labels != 255) & (labels >= 0)
                    correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                    total += valid_mask.sum().item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = correct / (total + 1e-10)

            if avg_val_loss < best_val_loss - early_stopping.min_delta:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())

            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)

            print(
                f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}"
            )

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"  -> Early stopping triggered at epoch {epoch+1}!")
                break
        else:
            print(f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    if val_loader is not None:
        model.load_state_dict(best_model_weights)
        print(f"* Restored best model weights (Val Loss: {best_val_loss:.4f})")

    plot_learning_curves(history, OUTPUTDIR, EXP_NUMBER)

    print(f"* Testing Prithvi-EO-V2-300 (UperNetDecoder), exp.{EXP_NUMBER}")
    model.eval()

    metric_calculator = SegmentationMetrics(num_classes=nclases, ignore_index=255)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cuda):
                outputs = model(inputs)

            metric_calculator.update(outputs, labels)

    global_metrics, per_class_metrics = metric_calculator.get_results()

    pAcc = global_metrics["pAcc"] * 100
    mAcc = global_metrics["mAcc"] * 100
    mIoU = global_metrics["mIoU"] * 100
    fwIoU = global_metrics["fwIoU"] * 100

    print(f"* Results exp.{EXP_NUMBER}:")
    print(f"  Pixel Accuracy (OA): {pAcc:.2f}%")
    print(f"  Mean Accuracy (AA):  {mAcc:.2f}%")
    print(f"  Mean IoU (mIoU):     {mIoU:.2f}%")
    print(f"  Freq Weighted IoU:   {fwIoU:.2f}%")

    class_aa = [0.0] + (per_class_metrics["acc"] * 100).tolist()

    ckpt_path = os.path.join(OUTPUTDIR, f"model_prithvi_exp{EXP_NUMBER}.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"* Model saved to {ckpt_path}")

    time_end = time.time()
    print(f"* Total time: {time_end - time_start:.0f} s")

    segmentation_stats = {
        "mIoU": mIoU,
        "fwIoU": fwIoU,
    }

    return (pAcc, mAcc, class_aa, segmentation_stats)