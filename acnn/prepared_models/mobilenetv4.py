import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import timm

from utils.dataloader import read_rawPV, read_grayscale_mask, generate_rgb_map
from utils.torch_utils import select_training_samples, HyperDataset, update_lr
from metrics.metrics import SegmentationMetrics

# ----------------
# DEFAULT GLOBALS
# ----------------
EXP = 1
EPOCHS = 30
SAMPLES = [0.05, 0.05] # [train_percentage, val_percentage]
BATCH = 256
SIZEX = 32
ADA = 3
AUM = 1
TEST = 1
OUTPUTDIR = "./results/"

def main(exp, image_path, gt_path, seg_path=None, centers_path=None, config=None):
    if config is not None:
        print("* Using config file")
        global EPOCHS, BATCH, SIZEX, AUM, ADA, TEST, OUTPUTDIR, SAMPLES
        exp = config.get("EXP_NUMBER", exp)
        EPOCHS = config.get("EPOCHS", EPOCHS)
        SAMPLES = config.get("SAMPLES", SAMPLES)
        BATCH = config.get("BATCH", BATCH)
        SIZEX = config.get("SIZEX", SIZEX)
        AUM = config.get("AUM", AUM)
        ADA = config.get("ADA", ADA)
        TEST = config.get("TEST", TEST)
        OUTPUTDIR = config.get("OUTPUTDIR", OUTPUTDIR)
        os.makedirs(OUTPUTDIR, exist_ok=True)

    print(f"* MobileNetV4 exp: {exp}")
    time_start = time.time()
    
    # 1. Device configuration
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    print(f"* Device: {device}")

    # 2. Load data (Memory Mapping)
    datos, H, V, B = read_rawPV(image_path, bit_depth=16)
    truth, H1, V1 = read_grayscale_mask(gt_path)
    
    # 3. Selection training, testing sets (Pixel level)
    train_idx, val_idx, test_idx, nclases, nclases_no_vacias = select_training_samples(
        truth, H, V, SIZEX, SIZEX, SAMPLES
    )
    
    dataset_train = HyperDataset(datos, truth, train_idx, H, V, SIZEX, SIZEX, bit_depth=16)
    dataset_test = HyperDataset(datos, truth, test_idx, H, V, SIZEX, SIZEX, bit_depth=16)
    
    train_loader = DataLoader(dataset_train, BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, BATCH, shuffle=False, num_workers=4, pin_memory=True)
    
    if len(val_idx) > 0:
        dataset_val = HyperDataset(datos, truth, val_idx, H, V, SIZEX, SIZEX, bit_depth=16)
        val_loader = DataLoader(dataset_val, BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Definition (timm MobileNetV4)
    # We set in_chans=4 for R-G-B-NIR and num_classes=24
    model = timm.create_model(
        "mobilenetv4_conv_medium",
        pretrained=True,
        in_chans=4,
        num_classes=nclases
    ).to(device)

    # 5. Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if ADA == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2, (5 * EPOCHS) // 6], gamma=0.1)
    elif ADA == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    elif ADA == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 6. Train the model
    print(f"* Training MobileNetV4, exp.{exp}")
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        if ADA > 1:
            scheduler.step()

        # Validation phase
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            if len(val_idx) > 0:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        preds = outputs.argmax(dim=-1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        
                print(f"  Epoch: {epoch:3d}/{EPOCHS}, Val Loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.4f}")
            else:
                print(f"  Epoch: {epoch:3d}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}")

    if TEST == 0:
        return correct / total

    # 7. Test the model using SegmentationMetrics
    print(f"* Testing MobileNetV4, exp.{exp}")
    model.eval()
    
    # Instance of your clean metric calculator (Labels are 0 to 23)
    metric_calculator = SegmentationMetrics(num_classes=nclases, ignore_index=255)
    
    output_map = np.zeros(H * V, dtype=np.uint8) 

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            metric_calculator.update(outputs, labels)
            
            # Map reconstruction (Optional but useful for visualization)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            # We map predictions back to 1-24 range for the RGB generator
            for i, idx in enumerate(test_idx[:len(preds)]):
                output_map[idx] = np.uint8(preds[i] + 1)
                
    # Fetch results
    global_metrics, per_class_metrics = metric_calculator.get_results()
    
    pAcc = global_metrics['pAcc'] * 100
    mAcc = global_metrics['mAcc'] * 100
    mIoU = global_metrics['mIoU'] * 100
    fwIoU = global_metrics['fwIoU'] * 100
    
    print(f"* Results exp.{exp}:")
    print(f"  Pixel Accuracy (OA): {pAcc:.2f}%")
    print(f"  Mean Accuracy (AA):  {mAcc:.2f}%")
    print(f"  Mean IoU (mIoU):     {mIoU:.2f}%")
    print(f"  Freq Weighted IoU:   {fwIoU:.2f}%")
    
    # We construct a 1-indexed list to match the old main.py expectation structure
    # class_aa index 0 is empty, index 1 corresponds to class 1, etc.
    class_aa = [0.0] + (per_class_metrics['acc'] * 100).tolist()

    # Generate classification map
    print("* Generating RGB classification map")
    img = generate_rgb_map(output_map, truth, H, V)
    img.save(os.path.join(OUTPUTDIR, f"output_rgb_mobilenetv4_exp{exp}.png"))
    
    torch.save(model.state_dict(), os.path.join(OUTPUTDIR, f"model_mobilenetv4_exp{exp}.ckpt"))

    time_end = time.time()
    print(f"* Total time: {time_end - time_start:.0f} s")

    # Returning exactly what main.py expects to not break the pipeline: 
    # (OA, AA, class_aa) + we inject mIoU instead of AD results for the last slot
    segmentation_stats = {
        "mIoU": mIoU,
        "fwIoU": fwIoU
    }
    
    return (pAcc, mAcc, class_aa, segmentation_stats)