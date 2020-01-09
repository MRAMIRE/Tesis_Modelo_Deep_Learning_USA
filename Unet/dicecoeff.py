import torch
import torch.nn as nn

def dice_coeff(pred,target):
  y_pred = pred.view(-1)
  m = nn.Sigmoid()
  y_real = target.view(-1)
  intersection = torch.sum(m(y_pred) * y_real)
  e = 0.00001
  return (2 * intersection + e) / (torch.sum(m(y_pred)) + torch.sum(y_real) + e)