import torch
import time
import math
import os
import re
import matplotlib.pyplot as plt
import sys

from package_visualization.display_images_ref_net_diff_checkerboard import display_images_ref_net_diff_checkerboard
from package_visualization.save_fig_on_harddrive import save_fig_on_harddrive
from package_geometry.define_scanner_and_fov_geometry import define_scanner_and_fov_geometry
from package_datasets.define_datasets import define_datasets
from package_networks import network_DeepMB
from package_networks.apply_network_forward_pass import apply_network_forward_pass
from package_utils.environment_check import environment_check
from package_utils.get_output_folders import get_output_folders
from package_utils.nifti import save_nifti

# Define TF log level (to hide messages such as "Successfully opened dynamic library libcudart.so.10.1")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

  # Root folder containing the results of all saved DeepMB experiments
  save_path = '???'

  # Experiment name (this ID uniquely identifies the network parameters and model)
  experiment_name = '???'

  # Import the parameters
  parameters_path = os.path.join(save_path, experiment_name, 'stored_parameters')
  sys.path.insert(0, parameters_path)
  from get_parameters import get_parameters

  # Determine what dataset will be inferred ['trn', 'val', 'tst']
  trn_val_tst = 'tst'

  # Specify whether each inferred image shall be saved as a nifti file
  SAVE_NII = True

  # Specify whether each inferred image shall be printed as a png
  SAVE_PNG = True

  # Get all parameters of a previous experiment
  p = get_parameters()

  # Reset parameters for inference mode
  p.BATCH_SIZE_TRN = 1
  p.BATCH_SIZE_VAL = 1
  p.BATCH_SIZE_TST = 1
  p.SAMPLES_LIMIT = math.inf

  # Optional: Change further parameters from the get_parameters-file used during training (e.g., "GPU_INDEX" or "DATASET_NAME_TST")
  # p.GPU_INDEX = 0
  # p.DATASET_NAME_TST = '???'

  # Sanity test
  use_cuda, device, num_workers = environment_check(p.DETERMINISTIC, p.RANDOM_SEED, p.GPU_INDEX)

  # Specify folders
  _, MODEL_CHECKPOINT_OUTPUT_DIR, _, _, \
  INFERENCE_OUTPUT_PATH_NII_TRN, INFERENCE_OUTPUT_PATH_PNG_TRN, \
  INFERENCE_OUTPUT_PATH_NII_VAL, INFERENCE_OUTPUT_PATH_PNG_VAL, \
  INFERENCE_OUTPUT_PATH_NII_TST, INFERENCE_OUTPUT_PATH_PNG_TST, \
  = get_output_folders(p.SAVE_PATH, p.EXPERIMENT_NAME, create_missing_folders=True)

  # Define the geometry corresponding to the imaging system and the field of view
  g = define_scanner_and_fov_geometry(p.SCANNER_ID)

  # Instantiate the Network
  network = network_DeepMB.Net(
    torch.from_numpy(g.transducer_pixel_distances).to(device), g.F_SAMPLE, g.SAMPLES_OFFSET, device, p, g)
  if use_cuda:
    network.cuda(device=device)
    network.float()

  # Load the trained network and set it in "eval" mode
  checkpoint = torch.load(os.path.join(MODEL_CHECKPOINT_OUTPUT_DIR, 'model_min_val_loss.pt'), map_location=device)
  network.load_state_dict(checkpoint['network_state_dict'])
  network.eval()

  # Load test dataset for inference
  loader_trn, nb_batch_trn, _, \
  loader_val, nb_batch_val, regex_print_val, \
  loader_tst, nb_batch_tst, regex_print_tst, \
  flip_trn, flip_val, flip_tst = define_datasets(p, num_workers)

  # Determine which dataloader and output folders shall be used in function of the processed dataset
  if trn_val_tst == 'trn':
    loader_set = loader_trn
    flipud = flip_trn
    nb_samples = nb_batch_trn
    INFERENCE_OUTPUT_PATH_NII = INFERENCE_OUTPUT_PATH_NII_TRN
    INFERENCE_OUTPUT_PATH_PNG = INFERENCE_OUTPUT_PATH_PNG_TRN
  elif trn_val_tst == 'val':
    loader_set = loader_val
    flipud = flip_val
    nb_samples = nb_batch_val
    INFERENCE_OUTPUT_PATH_NII = INFERENCE_OUTPUT_PATH_NII_VAL
    INFERENCE_OUTPUT_PATH_PNG = INFERENCE_OUTPUT_PATH_PNG_VAL
  elif trn_val_tst == 'tst':
    loader_set = loader_tst
    flipud = flip_tst
    nb_samples = nb_batch_tst
    INFERENCE_OUTPUT_PATH_NII = INFERENCE_OUTPUT_PATH_NII_TST
    INFERENCE_OUTPUT_PATH_PNG = INFERENCE_OUTPUT_PATH_PNG_TST

  # Specify a list of optional criteria to only process a subset, as for instance "['1495']", or "['p02', 'biceps']"
  REGEX_ADDITIONAL_NAME_FILTER = ['']

  # Enumerate the dataset
  with torch.no_grad():
    for id_batch, (input_sinogram, ima_ref, filename, _, sos) in enumerate(loader_set):

      # Check whether the name of the current sample matches (all of) the REGEX target(s)
      process_sample = True
      for target in REGEX_ADDITIONAL_NAME_FILTER:
        process_sample = process_sample and re.search(target, filename[0])
      if not process_sample:
        continue

      # Apply the network forward pass
      t0 = time.time()
      ima_net = apply_network_forward_pass(
        sinogram                               = input_sinogram.to(device).float(),
        sos                                    = sos.to(device).float(),
        network                                = network,
        TRANSFORM_TST_NETWORK_OUTPUT           = p.TRANSFORM_TST_NETWORK_OUTPUT,
        DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
      time_inference_ms = 1000 * (time.time() - t0)

      # Log the sample number and the inference time in the console
      print('(' + str(time_inference_ms) + ' ms)\t Sample ' + str(id_batch +1) + '/' + str(nb_samples))

      # Gather the network output back to CPU
      ima_net = ima_net.detach().cpu().numpy().squeeze() * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION

      # Print the reconstructed image as a PNG file
      if SAVE_PNG:
        fig = display_images_ref_net_diff_checkerboard(
          ima_ref.detach().cpu().numpy().squeeze() * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
          ima_net,
          filename[0], 'Deployed', trn_val_tst, flipud)
        save_fig_on_harddrive(fig, INFERENCE_OUTPUT_PATH_PNG, filename[0])
        plt.close(fig)

      # Save the reconstructed image as a nifti file
      if SAVE_NII:
        save_nifti(ima_net, INFERENCE_OUTPUT_PATH_NII, filename[0])
