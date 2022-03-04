import time
from package_utils.stopwatch import stopwatch


def print_epoch_loss(phase, epoch_id, loss, time_epoch, new_val_loss):

  # The symbol "UP_CLEAR_AND_RETURN" is used to print successive scores of the same {epoch, set} on the same console line
  # The effect is to overwrite the line written by the previous call to "print_batch_loss"
  # "Up one line, clear until end of line, return"
  UP_CLEAR_AND_RETURN = '\033[F\033[K'

  # The symbol "new_val_loss" is used to indicate if a new validation minimum has been reached
  if(phase == 'Val') and new_val_loss:
    message_new_val_loss = ' <--- [Min Val Loss]'
  else:
    message_new_val_loss = ''

  # Print the epoch loss
  print('{}[Epoch: {}, \t {}] Loss: {:.6f}\t (Stopwatch: {}){}'.format(
    UP_CLEAR_AND_RETURN,
    epoch_id,
    phase,
    loss,
    stopwatch(time.time(), time_epoch),
    message_new_val_loss))
