from torch.utils.tensorboard import SummaryWriter
import torch
import time
import math
import os
import torch.optim as optim
import matplotlib.pyplot as plt

from package_parameters.set_parameters import set_parameters
from package_parameters.create_get_parameters import create_get_parameters
from package_geometry.define_scanner_and_fov_geometry import define_scanner_and_fov_geometry
from package_datasets.define_datasets import define_datasets
from package_networks import network_DeepMB
from package_networks.custom_loss import custom_loss
from package_networks.custom_transform import custom_transform
from package_networks.apply_network_forward_pass import apply_network_forward_pass
from package_loggers.epoch_loss_accumulator import EpochLossAccumulator
from package_loggers.log_batch_console_tensorboard_harddrive import log_batch_to_console_tensorboard_harddrive
from package_loggers.log_epoch_console_tensorboard import log_epoch_console_tensorboard
from package_utils.environment_check import environment_check
from package_utils.get_output_folders import get_output_folders

# Define TF log level (to hide messages such as "Successfully opened dynamic library libcudart.so.10.1")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ==================================================================================
# === Main function: Initialize, then run Training, Validation, and Testing loop ===
# ==================================================================================
def main():

  # Set the experiment parameters (via the Git-ignored file "set_parameters.py")
  p = set_parameters()

  # Define the geometry corresponding to the imaging system and the field of view
  g = define_scanner_and_fov_geometry(p.SCANNER_ID)

  # Sanity test
  use_cuda, device, num_workers = environment_check(p.DETERMINISTIC, p.RANDOM_SEED, p.GPU_INDEX)

  # Get ima_net folders (ensuring that they exist)
  tensorboard_output_dir, model_checkpoint_output_dir, print_epoch_dir, stored_parameters_output_dir, \
  _, _, _, _, _, _ = get_output_folders(p.SAVE_PATH, p.EXPERIMENT_NAME, create_missing_folders=True)

  # Backup the experiment parameters
  create_get_parameters(stored_parameters_output_dir)

  # Indicate where to save the network checkpoint
  checkpoint_file_path_and_name = os.path.join(model_checkpoint_output_dir, 'model_min_val_loss.pt')

  # Indicate if the Trn/Val/Tst images must be be flipped vertically before display (in vivo images are upside-down)
  global flip_trn
  global flip_val
  global flip_tst

  # Get the dataloaders
  loader_trn, nb_batch_trn, _, \
  loader_val, nb_batch_val, regex_print_val, \
  loader_tst, nb_batch_tst, regex_print_tst, \
  flip_trn, flip_val, flip_tst = define_datasets(p, num_workers)

  # Overwrite the names of the VAL and TST images chosen for display, to ensure they are present in the dataset
  p.REGEX_PRINT_VAL = regex_print_val
  p.REGEX_PRINT_TST = regex_print_tst

  # Instantiate the Network
  network = network_DeepMB.Net(
    torch.from_numpy(g.transducer_pixel_distances).to(device), g.F_SAMPLE, g.SAMPLES_OFFSET, device, p, g)
  if use_cuda:
    network.cuda(device=device)
    network.float()

  # Starting epoch
  starting_epoch = 0

  # Optimizer and Scheduler
  if p.OPTIMIZER == 'SGD':
    optimizer = optim.SGD(network.parameters(), lr=p.LEARNING_RATE, momentum=p.MOMENTUM)
  elif p.OPTIMIZER == 'Adam':
    optimizer = optim.Adam(network.parameters(), lr=p.LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, p.SCHEDULER_STEP_SIZE, p.SCHEDULER_GAMMA)

  # Override the following starting parameters with a previously saved network checkpoint
  if p.RESUME_TRAINING:
    # Exact same network architecture, continues from the epoch with the min val loss
    checkpoint = torch.load(os.path.join(model_checkpoint_output_dir, 'model_min_val_loss.pt'), map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    starting_epoch = checkpoint['epoch']
  elif p.USE_PRETRAINING:
    # Possibly different network architecture, starts at epoch zero
    checkpoint = torch.load(os.path.join(model_checkpoint_output_dir, 'model_min_val_loss.pt'), map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'], strict=False)
    # Here is an example where the parameters related to the signal network are frozen
    for name, param in network.named_parameters():
      if name[0:7] == 'net_sig':
        param.requires_grad = False

  # Memorize the best val loss so far
  global min_val_loss
  min_val_loss = math.inf

  # Indicate if a new minimum has been reached with the validation loss
  global new_val_loss
  new_val_loss = False

  # Memorize the test loss of the model with the best val loss so far
  global last_test_loss
  last_test_loss = 0.0

  # Instantiate the Tensorboard writer to log the results
  writer = SummaryWriter(tensorboard_output_dir)

  # Run the training, validation, and testing loop
  for id_epoch in range(starting_epoch, p.NB_EPOCH):
    trn(p, network, optimizer, scheduler, loader_trn, nb_batch_trn, id_epoch, device, print_epoch_dir, writer)
    val(p, network, optimizer, scheduler, loader_val, nb_batch_val, id_epoch, device, print_epoch_dir, writer, checkpoint_file_path_and_name)
    tst(p, network, loader_tst, nb_batch_tst, id_epoch, device, print_epoch_dir, writer)
    plt.close('all')

  # Close the Tensorboard writer
  writer.close()


# ================================================================
# === Training routine ===========================================
# ================================================================
def trn(p, network, optimizer, scheduler, loader_trn, nb_batch_trn, id_epoch, device, print_epoch_dir, writer):

  global flip_trn

  network.train()
  trn_loss_accumulator = EpochLossAccumulator()
  time_epoch = time.time()

  for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch, _, sos_rec_batch) in enumerate(loader_trn):
    ima_ref_batch = custom_transform(
      ima_ref_batch.to(device).float(), p.TRANSFORM_TRN_VAL_TARGETS, p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
    # Reset gradients to zero
    optimizer.zero_grad()
    # Forward pass with the deep neural network
    ima_net_batch = apply_network_forward_pass(
      sinogram = sinogram_batch.to(device).float(),
      sos      = sos_rec_batch.to(device).float(),
      network  = network)
    # Calculate loss, propagate gradients back through the network, and optimize
    loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
    loss.backward()
    optimizer.step()
    # Keep track of the calculated loss
    batch_size = sinogram_batch.shape[0]
    trn_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
    # Log the current batch
    log_batch_to_console_tensorboard_harddrive(
      id_epoch, 'Trn', trn_loss_accumulator, time_epoch, id_batch, nb_batch_trn, ima_ref_batch, ima_net_batch,
      file_name_batch, writer, flip_trn, p, print_epoch_dir)

  # Log the current epoch
  current_train_loss = trn_loss_accumulator.get_epoch_loss()
  log_epoch_console_tensorboard(
    writer, 'Trn', '0_training_loss', id_epoch, time_epoch, current_train_loss, new_val_loss)

  # Learning rate evolution
  scheduler.step()


# ================================================================
# === Validation routine =========================================
# ================================================================
def val(p, network, optimizer, scheduler, loader_val, nb_batch_val, id_epoch, device, print_epoch_dir, writer, checkpoint_file_path_and_name):

  global min_val_loss
  global new_val_loss
  global flip_val

  network.eval()
  val_loss_accumulator = EpochLossAccumulator()
  time_epoch = time.time()

  with torch.no_grad():
    for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch, _, sos_rec_batch) in enumerate(loader_val):
      ima_ref_batch = custom_transform(
        ima_ref_batch.to(device).float(), p.TRANSFORM_TRN_VAL_TARGETS, p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
      # Forward pass with the deep neural network
      ima_net_batch = apply_network_forward_pass(
        sinogram = sinogram_batch.to(device).float(),
        sos      = sos_rec_batch.to(device).float(),
        network  = network)
      # Loss
      loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
      # Keep track of the calculated loss
      batch_size = sinogram_batch.shape[0]
      val_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
      # Log the current batch
      log_batch_to_console_tensorboard_harddrive(
        id_epoch, 'Val', val_loss_accumulator, time_epoch, id_batch, nb_batch_val, ima_ref_batch, ima_net_batch,
        file_name_batch, writer, flip_val, p, print_epoch_dir)

  # Save current network checkpoint if new minimal validation loss is found
  current_val_loss = val_loss_accumulator.get_epoch_loss()
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    new_val_loss = True
    torch.save({
      'epoch': id_epoch +1,
      'network_state_dict': network.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict()},
      checkpoint_file_path_and_name)
  else:
    new_val_loss = False

  # Log the current epoch
  log_epoch_console_tensorboard(
    writer, 'Val', '1_validation_loss', id_epoch, time_epoch, current_val_loss, new_val_loss)


# ================================================================
# === Testing routine ============================================
# ================================================================
def tst(p, network, loader_tst, nb_batch_tst, id_epoch, device, print_epoch_dir, writer):

  global new_val_loss
  global last_test_loss
  global flip_tst

  time_epoch = time.time()

  if new_val_loss:
    network.eval()
    tst_loss_accumulator = EpochLossAccumulator()
    with torch.no_grad():
      for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch, _, sos_rec_batch) in enumerate(loader_tst):
        # Forward pass with the deep neural network
        ima_net_batch = apply_network_forward_pass(
          sinogram                               = sinogram_batch.to(device).float(),
          sos                                    = sos_rec_batch.to(device).float(),
          network                                = network,
          TRANSFORM_TST_NETWORK_OUTPUT           = p.TRANSFORM_TST_NETWORK_OUTPUT,
          DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
        # Loss
        ima_ref_batch = ima_ref_batch.to(device).float()
        loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
        # Keep track of the calculated loss
        batch_size = sinogram_batch.shape[0]
        tst_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
        # Log the current batch
        log_batch_to_console_tensorboard_harddrive(
          id_epoch, 'Tst', tst_loss_accumulator, time_epoch, id_batch, nb_batch_tst, ima_ref_batch, ima_net_batch,
          file_name_batch, writer, flip_tst, p, print_epoch_dir)

    # Keep track of last_test_loss
    printed_test_loss = tst_loss_accumulator.get_epoch_loss()
    last_test_loss = printed_test_loss

  else:
    # This dummy text in the console will not be seen, but is a necessary placeholder for the log display
    print('(The test set is not processed)')
    printed_test_loss = last_test_loss

  # Log the current epoch
  log_epoch_console_tensorboard(
    writer, 'Tst', '2_testing_loss', id_epoch, time_epoch, printed_test_loss, new_val_loss)


# ================================================================
# === Entry point ================================================
# ================================================================
if __name__ == '__main__':

  # Define required global variables at module level
  global flip_trn
  global flip_val
  global flip_tst
  global min_val_loss
  global new_val_loss
  global last_test_loss

  # Call main function
  main()
