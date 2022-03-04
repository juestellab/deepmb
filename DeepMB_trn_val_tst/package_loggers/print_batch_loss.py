import time
import numpy as np
from package_utils.stopwatch import stopwatch


def print_batch_loss(epoch_id, phase, epoch_loss_accumulator, time_epoch, batch_id, nb_batches):

  # The symbol "UP_AND_CLEAR" is used to print successive scores of the same {epoch, set} on the same console line
  if batch_id == 0:
    # The first log of an epoch shall not overwrite the console line above
    UP_AND_CLEAR = ''
  else:
    # Printing "UP_AND_CLEAR" in the log will overwrite the line written by the previous call to "print_batch_loss"
    # "Up one line, clear until end of line"
    UP_AND_CLEAR = '\033[F\033[K'

  # The symbol "NEW_LINE" is used to mark the succession between different epochs by a blank line
  if(batch_id == 0) and (epoch_id > 0) and (phase == 'Trn'):
    # A new line is needed to separate the log when entering within a new epoch
    NEW_LINE = '\n'
  else:
    # No separation space is required
    NEW_LINE = ''

  # Choice to display "ID +1", to indicate the fractional progression instead of the actual batch ID
  one_indexed_offset = 1

  # Print the batch loss
  print('{}{}[Epoch: {},\t {}] Batch: {}/{} ({}%)\t Loss: {:.6f}\t (Stopwatch: {})'.format(
    UP_AND_CLEAR,
    NEW_LINE,
    epoch_id,
    phase,
    batch_id + one_indexed_offset,
    nb_batches,
    np.rint(100 * (batch_id + one_indexed_offset) / nb_batches).astype(np.int),
    epoch_loss_accumulator.get_and_reset_sub_loss(),
    stopwatch(time.time(), time_epoch)))
