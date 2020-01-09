import torch
import torch.nn as nn

def cross_entropy(prediction,target):
  y_pred = prediction.view(-1)
  y_real = target.view(-1)
  m = nn.Sigmoid()
  loss = nn.BCELoss()
  return loss(m(y_pred), y_real)