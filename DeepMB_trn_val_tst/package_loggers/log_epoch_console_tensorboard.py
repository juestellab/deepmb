from .print_epoch_loss import print_epoch_loss


def log_epoch_console_tensorboard(
  writer, trn_val_tst_short_namecode, trn_val_tst_long_namecode, id_epoch, time_epoch, loss, new_val_loss):

  # Print log to console
  print_epoch_loss(trn_val_tst_short_namecode, id_epoch, loss, time_epoch, new_val_loss)

  # Write log to Tensorboard
  writer.add_scalar(trn_val_tst_long_namecode, loss, id_epoch)
  writer.flush()
