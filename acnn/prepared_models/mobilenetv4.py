import os
import time
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from utils.torch_utils import DenseQuarterDataset, EarlyStopping
from utils.metrics import SegmentationMetrics
from utils.general_utils import plot_learning_curves

def main(EXP_NUMBER, config):
    print(f"* MobileNetV4 exp: {EXP_NUMBER}")
    
    # Directory and parameter configuration from config.json
    TRAIN_DIR = config.get("train_dir")
    TRAIN_GT_DIR = config.get("train_gt_dir")
    TEST_DIR = config.get("test_dir")
    TEST_GT_DIR = config.get("test_gt_dir")

    # SAMPLES format: [train_proportion, val_proportion]
    SAMPLES = config.get("SAMPLES", [0.9, 0.1])  # Default: 90% train, 10% validation
    EPOCHS = config.get("EPOCHS", 50)
    # batch size must be tiny (1, 2, 4) due to the large quarter sizes
    BATCH_SIZE = config.get("BATCH_SIZE", 2)
    PATIENCE = config.get("PATIENCE", 5)
    ADA = config.get("ADA", 3)
    OUTPUTDIR = config.get("OUTPUTDIR", "./results/")
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # Config logging
    print(f"* Train Directory: {TRAIN_DIR}")
    print(f"* Train GT Directory: {TRAIN_GT_DIR}")
    print(f"* Test Directory: {TEST_DIR}")
    print(f"* Test GT Directory: {TEST_GT_DIR}")
    print(f"* Samples proportions: {SAMPLES}")
    print(f"* Epochs: {EPOCHS}")
    print(f"* Batch Size: {BATCH_SIZE}")
    print(f"* Early Stopping Patience: {PATIENCE}")
    print(f"* ADA (LR Scheduler): {ADA}")
    print(f"* Output Directory: {OUTPUTDIR}")

    time_start = time.time()
    
    # Device configuration
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    print(f"* Device: {device}")

    # Prepare files
    all_train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.raw")))
    all_train_masks = sorted(glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.raw")))
    test_masks = sorted(glob.glob(os.path.join(TEST_GT_DIR, "*.png")))
    
    if len(all_train_images) == 0 or len(test_images) == 0:
        raise ValueError("No .raw images found. Please check config.json directories.")

    H, V, nclases = 6908, 7300, 24

    # Train / Val Split based on SAMPLES proportion
    # Combine and shuffle to ensure random distribution of images
    combined_train = list(zip(all_train_images, all_train_masks))
    random.seed(42) # Fixed seed for reproducibility across experiments
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

    print(f"* Split -> Train images: {len(train_images)} | Val images: {len(val_images)} | Test images: {len(test_images)}")

    # Instantiate Datasets and DataLoaders. I/O will be heavy but sequential.
    train_dataset = DenseQuarterDataset(train_images, train_masks, H, V, transform_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    if len(val_images) > 0:
        val_dataset = DenseQuarterDataset(val_images, val_masks, H, V, transform_flag=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    test_dataset = DenseQuarterDataset(test_images, test_masks, H, V, transform_flag=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Definition (SMP FPN + timm MobileNetV4)
    # Using 'tu-' instructs SMP to use timm, applying features_only=True automatically.
    model = smp.FPN(
        encoder_name="tu-mobilenetv4_conv_medium",
        encoder_weights=None,      # pretrained=False
        in_channels=4,             # RGB + NIR
        classes=nclases,           # Automatic classification head included in FPN
    ).to(device)

    # Loss, optimizer, scheduler, and Early Stopping
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if ADA == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2, (5 * EPOCHS) // 6], gamma=0.1)
    elif ADA == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    elif ADA == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.0001)

    # Scaler for mixed precision training (if CUDA is available)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    # Train the model
    print(f"* Training FPN-MobileNetV4, exp.{EXP_NUMBER}")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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
        history['train_loss'].append(avg_train_loss)
        
        if ADA > 1:
            scheduler.step()

        # Dense validation phase
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

            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(avg_val_acc)

            print(f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")
            
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"  -> Early stopping triggered at epoch {epoch+1}!")
                break
        else:
            print(f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    plot_learning_curves(history, OUTPUTDIR, EXP_NUMBER)

    # Test the model
    print(f"* Testing MobileNetV4, exp.{EXP_NUMBER}")
    model.eval()
    
    metric_calculator = SegmentationMetrics(num_classes=nclases, ignore_index=255)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cuda):
                outputs = model(inputs)
                
            metric_calculator.update(outputs, labels)
                
    # Fetch results
    global_metrics, per_class_metrics = metric_calculator.get_results()
    
    pAcc = global_metrics['pAcc'] * 100
    mAcc = global_metrics['mAcc'] * 100
    mIoU = global_metrics['mIoU'] * 100
    fwIoU = global_metrics['fwIoU'] * 100
    
    print(f"* Results exp.{EXP_NUMBER}:")
    print(f"  Pixel Accuracy (OA): {pAcc:.2f}%")
    print(f"  Mean Accuracy (AA):  {mAcc:.2f}%")
    print(f"  Mean IoU (mIoU):     {mIoU:.2f}%")
    print(f"  Freq Weighted IoU:   {fwIoU:.2f}%")
    
    class_aa = [0.0] + (per_class_metrics['acc'] * 100).tolist()
    
    torch.save(model.state_dict(), os.path.join(OUTPUTDIR, f"model_mobilenetv4_exp{EXP_NUMBER}.ckpt"))
    print(f"* Model saved to {os.path.join(OUTPUTDIR, f'model_mobilenetv4_exp{EXP_NUMBER}.ckpt')}")

    time_end = time.time()
    print(f"* Total time: {time_end - time_start:.0f} s")

    segmentation_stats = {
        "mIoU": mIoU,
        "fwIoU": fwIoU
    }
    
    return (pAcc, mAcc, class_aa, segmentation_stats)