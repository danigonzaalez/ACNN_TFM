import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from utils.torch_utils import MultiImageDataset, build_global_samples, EarlyStopping, update_lr
from utils.metrics import SegmentationMetrics
from utils.general_utils import plot_learning_curves, print_global_sampling_summary

def main(EXP_NUMBER, config):
    print(f"* MobileNetV4 exp: {EXP_NUMBER}")
    
    # 1. Directory and parameter configuration from config.json
    TRAIN_DIR = config.get("train_dir")
    TRAIN_GT_DIR = config.get("train_gt_dir")
    TEST_DIR = config.get("test_dir")
    TEST_GT_DIR = config.get("test_gt_dir")

    print(f"* Train Directory: {TRAIN_DIR}")
    print(f"* Train GT Directory: {TRAIN_GT_DIR}")
    print(f"* Test Directory: {TEST_DIR}")
    print(f"* Test GT Directory: {TEST_GT_DIR}")
    
    # SAMPLES format: [train_samples_per_image, val_samples_per_image]
    SAMPLES = config.get("SAMPLES", [2000, 500]) 
    TEST_SAMPLES = config.get("TEST_SAMPLES", 2000)
    EPOCHS = config.get("EPOCHS", 50)
    BATCH_SIZE = config.get("BATCH_SIZE", 256)
    PATIENCE = config.get("PATIENCE", 5)
    SIZEX = config.get("SIZEX", 32)
    ADA = config.get("ADA", 3)
    OUTPUTDIR = config.get("OUTPUTDIR", "./results/")
    os.makedirs(OUTPUTDIR, exist_ok=True)

    time_start = time.time()
    
    # Device configuration
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    print(f"* Device: {device}")

    # 2. Prepare file lists
    # Ensure images and masks are correctly paired by alphabetical order
    train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.raw")))
    train_masks = sorted(glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.raw")))
    test_masks = sorted(glob.glob(os.path.join(TEST_GT_DIR, "*.png")))
    
    if len(train_images) == 0 or len(test_images) == 0:
        raise ValueError("No .raw images found in the specified directories. Please check config.json.")

    H, V = 6908, 7300
    nclases = 24 

    # 3. Build global indices (spatial subsampling)
    train_idx, val_idx, train_stats = build_global_samples(train_masks, H, V, SIZEX, SIZEX, SAMPLES[0], SAMPLES[1], dataset_name="Train/Val")
    test_idx, _, test_stats = build_global_samples(test_masks, H, V, SIZEX, SIZEX, TEST_SAMPLES, 0, dataset_name="Test")

    # Print the summary table for global sampling
    print_global_sampling_summary(train_stats, test_stats, train_idx, val_idx, test_idx, nclases)

    # 4. Instantiate Datasets and DataLoaders
    # train dataset uses data augmentation (transform_flag=True), val and test do not.
    train_dataset = MultiImageDataset(train_images, train_idx, H, V, SIZEX, SIZEX, transform_flag=True)
    val_dataset = MultiImageDataset(train_images, val_idx, H, V, SIZEX, SIZEX, transform_flag=False)
    test_dataset = MultiImageDataset(test_images, test_idx, H, V, SIZEX, SIZEX, transform_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model Definition
    model = timm.create_model(
        "mobilenetv4_conv_medium",
        pretrained=False,
        in_chans=4,
        num_classes=nclases
    ).to(device)

    # 6. Loss, optimizer, scheduler, and Early Stopping
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if ADA == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2, (5 * EPOCHS) // 6], gamma=0.1)
    elif ADA == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    elif ADA == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.0001)

    # 7. Train the model
    print(f"* Training MobileNetV4, exp.{EXP_NUMBER}")
    
    # Dictionary to store metrics for plotting
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Progress bar iterating over batches
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Calculate average train loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if ADA > 1:
            scheduler.step()

        # Validation phase
        if len(val_idx) > 0:
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = correct / total

            # Save validation metrics
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(avg_val_acc)

            print(f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")
            
            # Check Early Stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"  -> Early stopping triggered at epoch {epoch+1}! Model has converged.")
                break
        else:
            print(f"  Epoch: {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    # Generate and save learning curves plot
    print("* Generating learning curves plot...")
    plot_learning_curves(history, OUTPUTDIR, EXP_NUMBER)

    # 8. Test the model
    print(f"* Testing MobileNetV4, exp.{EXP_NUMBER}")
    model.eval()
    
    metric_calculator = SegmentationMetrics(num_classes=nclases, ignore_index=255)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
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

    time_end = time.time()
    print(f"* Total time: {time_end - time_start:.0f} s")

    segmentation_stats = {
        "mIoU": mIoU,
        "fwIoU": fwIoU
    }
    
    return (pAcc, mAcc, class_aa, segmentation_stats)