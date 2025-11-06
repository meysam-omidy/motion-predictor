import torch
from torch import nn

class LossFunction(nn.Module):
    def forward(self, targets, preds):
        bb1 = targets.unsqueeze(2)        # (B, N, 1, 4)
        bb2 = preds.unsqueeze(1)      # (B, 1, N, 4)
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
        ious = inter / union
        ious = torch.diagonal(ious, dim1=1, dim2=2)
        loss1 = nn.functional.l1_loss(ious, torch.ones_like(ious))
        loss2 = nn.functional.smooth_l1_loss(targets, preds)
        # return loss2
        return loss1 + loss2