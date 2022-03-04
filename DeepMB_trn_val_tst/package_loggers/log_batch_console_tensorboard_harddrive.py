from package_visualization.save_fig_on_harddrive import save_fig_on_harddrive
from package_visualization.display_images_ref_net_diff_checkerboard import display_images_ref_net_diff_checkerboard
from .print_batch_loss import print_batch_loss


def log_batch_to_console_tensorboard_harddrive(
  id_epoch, trn_val_tst, loss_accumulator, time_epoch, id_batch, nb_batch, ima_ref_batch, ima_net_batch,
  file_name_batch, writer, flip, p, print_epoch_dir):

  # Log to console
  print_batch_loss(id_epoch, trn_val_tst, loss_accumulator, time_epoch, id_batch, nb_batch)

  # Log images to tensorboard and harddrive (if applicable for the current batch)
  if trn_val_tst == 'Trn':
    if id_batch == p.ID_PRINT_TRN[0]: # Tensorboard
      fig_tensorboard = display_images_ref_net_diff_checkerboard(
        ima_ref_batch.detach().cpu().numpy()[0][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        ima_net_batch.detach().cpu().numpy()[0][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        file_name_batch[0], id_epoch, trn_val_tst, flip)
      writer.add_figure('0_' + trn_val_tst, fig_tensorboard, global_step=id_epoch)

    if id_batch == p.ID_PRINT_TRN[1]: # Harddrive
      fig_harddrive = display_images_ref_net_diff_checkerboard(
        ima_ref_batch.detach().cpu().numpy()[0][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        ima_net_batch.detach().cpu().numpy()[0][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        file_name_batch[0], id_epoch, trn_val_tst, flip)
      save_fig_on_harddrive(
        fig_harddrive, print_epoch_dir, file_name_batch[0], trn_val_tst=trn_val_tst, id_epoch=id_epoch)

  else: # (trn_val_tst == 'Val' or 'Tst')
    # Get the two regex for the current phase
    if trn_val_tst == 'Val':
      regex_tensorboard = p.REGEX_PRINT_VAL[0]
      regex_harddrive = p.REGEX_PRINT_VAL[1]
    else:
      regex_tensorboard = p.REGEX_PRINT_TST[0]
      regex_harddrive = p.REGEX_PRINT_TST[1]

    # Check if the two regex match with the current file_name_batch
    sample_index_tensorboard = file_name_batch.index(regex_tensorboard) if regex_tensorboard in file_name_batch else -1
    sample_index_harddrive = file_name_batch.index(regex_harddrive) if regex_harddrive in file_name_batch else -1

    # Write to tensorboard and harddrive, respectively
    if sample_index_tensorboard > -1:
      fig_tensorboard = display_images_ref_net_diff_checkerboard(
        ima_ref_batch.detach().cpu().numpy()[sample_index_tensorboard][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        ima_net_batch.detach().cpu().numpy()[sample_index_tensorboard][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        file_name_batch[sample_index_tensorboard], id_epoch, trn_val_tst, flip)
      prefix = '1_' if trn_val_tst == 'Val' else '2_'
      writer.add_figure(
        prefix + trn_val_tst + ': ' + file_name_batch[sample_index_tensorboard], fig_tensorboard, global_step=id_epoch)

    if sample_index_harddrive > -1:
      fig_harddrive = display_images_ref_net_diff_checkerboard(
        ima_ref_batch.detach().cpu().numpy()[sample_index_harddrive][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        ima_net_batch.detach().cpu().numpy()[sample_index_harddrive][0] * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
        file_name_batch[sample_index_harddrive], id_epoch, trn_val_tst, flip)
      save_fig_on_harddrive(
        fig_harddrive, print_epoch_dir, file_name_batch[sample_index_harddrive], trn_val_tst=trn_val_tst,
        id_epoch=id_epoch)
