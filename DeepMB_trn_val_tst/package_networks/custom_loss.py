import torch
import torch.nn.functional as F


def custom_loss(net_out, target_ima, loss_nickname):

  # Mean Square Error (L2)
  if loss_nickname == 'MSE':
    loss = F.mse_loss(net_out, target_ima)

  # Mean Absolute Error (L1)
  elif loss_nickname == 'MAE':
    loss = F.l1_loss(net_out, target_ima)

  # Huber
  elif loss_nickname == 'HUBER':
    loss = F.smooth_l1_loss(net_out, target_ima, beta=0.1)

  return loss
