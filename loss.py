import torch
import math
from torch import nn

def MF1_criterion(result, masks):
    MD1 = torch.mean((result - masks)**2 * masks)
    FA1 = torch.mean((result - masks)**2 * (1 - masks))
    MF_loss1 = MD1 * 10 + FA1
    return MD1, FA1, MF_loss1


def SoftIoULoss(pred, target):
    smooth = 1
    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
    loss = 1 - loss.mean()
    return loss
