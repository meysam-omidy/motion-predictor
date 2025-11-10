import torch
from torch import nn

class LossFunction(nn.Module):
    def __init__(self, loss1_coeff=1, loss2_coeff=1, loss3_coeff=1, loss4_coeff=0.5, use_motion_features=True):
        super().__init__()
        self.loss1_coeff = loss1_coeff  # IOU loss
        self.loss2_coeff = loss2_coeff  # Bbox regression loss
        self.loss3_coeff = loss3_coeff  # Confidence loss
        self.loss4_coeff = loss4_coeff  # Motion features loss (velocity/acceleration)
        self.use_motion_features = use_motion_features


    def forward(self, targets, preds):
        target_boxes = targets[:, :, :4]
        pred_boxes = preds[:, :, :4]
        
        # Compute IOU loss
        bb1 = target_boxes.unsqueeze(2)        # (B, N, 1, 4)
        bb2 = pred_boxes.unsqueeze(1)      # (B, 1, N, 4)
        xx1 = torch.maximum(bb1[..., 0], bb2[..., 0])
        yy1 = torch.maximum(bb1[..., 1], bb2[..., 1])
        xx2 = torch.minimum(bb1[..., 2], bb2[..., 2])
        yy2 = torch.minimum(bb1[..., 3], bb2[..., 3])
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        area1 = (bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])
        area2 = (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1])
        union = area1 + area2 - inter
        ious = inter / (union + 1e-7)  # Add epsilon to prevent division by zero
        ious = torch.diagonal(ious, dim1=1, dim2=2)
        
        loss1 = nn.functional.smooth_l1_loss(torch.ones_like(ious), ious)
        loss2 = nn.functional.smooth_l1_loss(target_boxes, pred_boxes)
        
        # Handle both old (5-dim) and new (13-dim) formats
        if self.use_motion_features and targets.size(-1) == 13:
            # New format: 12 motion features + 1 confidence
            target_cs = targets[:, :, -1]
            pred_cs = preds[:, :, -1]
            loss3 = nn.functional.smooth_l1_loss(target_cs, pred_cs)
            
            # Add loss for velocity and acceleration predictions
            target_motion = targets[:, :, 4:12]  # velocity and acceleration
            pred_motion = preds[:, :, 4:12]
            loss4 = nn.functional.smooth_l1_loss(target_motion, pred_motion)
            
            return (self.loss1_coeff * loss1 + 
                    self.loss2_coeff * loss2 + 
                    self.loss3_coeff * loss3 + 
                    self.loss4_coeff * loss4)
        else:
            # Old format: 4 bbox coords + 1 confidence
            target_cs = targets[:, :, 4]
            pred_cs = preds[:, :, 4]
            loss3 = nn.functional.smooth_l1_loss(target_cs, pred_cs)
            
            return self.loss1_coeff * loss1 + self.loss2_coeff * loss2 + self.loss3_coeff * loss3