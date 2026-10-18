import numpy as np
import torch

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=0):
        self.num_classes      = num_classes
        self.ignore_index     = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        preds  = torch.argmax(preds, dim=1)
        preds  = preds.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        valid  = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_classes)
        preds, labels = preds[valid], labels[valid]
        bc = np.bincount(self.num_classes * labels + preds,
                         minlength=self.num_classes**2)
        self.confusion_matrix += bc.reshape((self.num_classes, self.num_classes))

    def get_results(self):
        hist  = self.confusion_matrix
        tp    = np.diag(hist)
        gt    = hist.sum(axis=1)
        pr    = hist.sum(axis=0)
        pAcc  = np.sum(tp) / (np.sum(hist) + 1e-10)
        cacc  = tp / (gt + 1e-10)
        mAcc  = np.mean(cacc[gt > 0])
        union = gt + pr - tp
        iou   = tp / (union + 1e-10)
        mIoU  = np.mean(iou[gt > 0])
        freq  = gt / (np.sum(hist) + 1e-10)
        fwIoU = np.sum(freq[gt > 0] * iou[gt > 0])
        return (
            {'pAcc': pAcc, 'mAcc': mAcc, 'mIoU': mIoU, 'fwIoU': fwIoU},
            {'iou': np.where(gt > 0, iou, 0.0),
             'acc': np.where(gt > 0, cacc, 0.0)},
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))



# EJEMPLO DE USO
# metric_calculator  = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_VALUE)

# model.eval()
# with torch.no_grad():
#     for images, masks in test_loader:
#         images, masks = images.to(device), masks.to(device)
#         outputs       = model(images)
#         metric_calculator.update(outputs, masks)

# global_metrics, per_class_metrics = metric_calculator.get_results()

# pAcc  = global_metrics['pAcc']
# mAcc  = global_metrics['mAcc']
# mIoU  = global_metrics['mIoU']
# fwIoU = global_metrics['fwIoU']

# print(f"Pixel Accuracy (pAcc):         {pAcc:.4f}")
# print(f"Mean Class Accuracy (mAcc):    {mAcc:.4f}")
# print(f"Mean IoU (mIoU):               {mIoU:.4f}")
# print(f"Frequency Weighted IoU (fwIoU):{fwIoU:.4f}")
# print("  Per-class IoU: " +
#         "  ".join(f"c{i}={v:.4f}" for i, v in enumerate(per_class_metrics['iou'])))
# print("  Per-class Acc: " +
#         "  ".join(f"c{i}={v:.4f}" for i, v in enumerate(per_class_metrics['acc'])))
# print("-" * 60)