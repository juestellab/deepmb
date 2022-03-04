import time
from package_utils import stopwatch


class EpochLossAccumulator:

  def __init__(self):
    self.loss_all = 0.0   # Cumulative loss of the current epoch
    self.loss_sub = 0.0   # Cumulative loss of the current sub-epoch (cf. "LOG_STEP")
    self.nb_items_all = 0 # Number of total samples for the current epoch
    self.nb_items_sub = 0 # Number of total samples for the current sub-epoch (cf. "LOG_STEP")

  def update_losses(self, batch_size, loss_item):
    self.nb_items_all += batch_size
    self.nb_items_sub += batch_size
    self.loss_all += loss_item
    self.loss_sub += loss_item

  def get_epoch_loss(self):
    if self.nb_items_all == 0: # (empty set)
      return 0
    else:
      return self.loss_all / self.nb_items_all

  def get_and_reset_sub_loss(self):
    if self.nb_items_sub == 0: # (empty set)
      sub_loss = 0
    else:
      sub_loss = self.loss_sub / self.nb_items_sub
    self.nb_items_sub = 0
    self.loss_sub = 0.0
    return sub_loss
