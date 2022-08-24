import os
import torch
import numpy as np
from medpy.io import load
from torchvision import transforms
import sys

from package_geometry.define_scanner_and_fov_geometry import define_scanner_and_fov_geometry
from package_networks import network_DeepMB
from package_networks.apply_network_forward_pass import apply_network_forward_pass
from package_utils.environment_check import environment_check
from package_visualization.showcase_demo_DeepMB_ifr import showcase_demo_DeepMB_ifr

if __name__ == '__main__':

  ### Easy-access parameters that disregard those obtained by "get_parameters.py"
  # Root folder containing the results of all saved DeepMB experiments
  save_path = '???'
  # Experiment name (unique identifier of the network parameters "get_parameters.py" and model "model_min_val_loss.pt")
  experiment_name = '???'
  # Absolute path to the sinogram input file
  sinogram_path = '???'
  # Name of the sinogram input file (including extension, e.g., ".nii")
  sinogram_filename = '???'
  # Absolute path to the reference model-based reconstruction of the input sinogram (if available, else please use an empty string '')
  reference_image_path = '???'
  # Speed of sound [m/s]
  sos = 1540
  # Upside-down flip shall be applied to in-vivo images
  flipud = True

  # Import the 'get_parameters' module from the specified experiment
  parameters_path = os.path.join(save_path, experiment_name, 'stored_parameters')
  sys.path.insert(0, parameters_path)
  from get_parameters import get_parameters

  # Get the experiment parameters
  p = get_parameters()

  # Define the geometry corresponding to the imaging system and the field of view
  g = define_scanner_and_fov_geometry(p.SCANNER_ID)

  # Sanity test
  use_cuda, device, _ = environment_check(p.DETERMINISTIC, p.RANDOM_SEED, p.GPU_INDEX)

  # Instantiate the Network
  network = network_DeepMB.Net(
    torch.from_numpy(g.transducer_pixel_distances).to(device), g.F_SAMPLE, g.SAMPLES_OFFSET, device, p, g)
  if use_cuda:
    network.cuda(device=device)
    network.float()

  # Load the trained network and set it in "eval" mode
  network_checkpoint_path_and_filename = os.path.join(
    save_path, experiment_name, 'model_checkpoint', 'model_min_val_loss.pt')
  checkpoint = torch.load(network_checkpoint_path_and_filename, map_location=device)
  network.load_state_dict(checkpoint['network_state_dict'])
  network.eval()

  # Load the sinogram
  input_sinogram, _ = load(os.path.join(sinogram_path, sinogram_filename))

  # Load the model-based reference image (if available)
  if not reference_image_path == '':
    reference_image, _ = load(os.path.join(reference_image_path, sinogram_filename))
  else:
    reference_image = None

  # Define the Tensor transformation that is applied to the input sinogram
  transform_input = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION])])

  # Tensor-transform the sinogram as well as the SoS value
  input_sinogram = transform_input(input_sinogram).unsqueeze(0)
  sos = torch.tensor(float(sos)).unsqueeze(0)

  # Apply the network forward pass
  with torch.no_grad():
    output_image = apply_network_forward_pass(
      sinogram                               = input_sinogram.to(device).float(),
      sos                                    = sos.to(device).float(),
      network                                = network,
      TRANSFORM_TST_NETWORK_OUTPUT           = p.TRANSFORM_TST_NETWORK_OUTPUT,
      DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

    # Synchronous timing for info (the previous call to "apply_network_forward_pass" was used as the warm-up procedure)
    time_start = torch.cuda.Event(enable_timing = True)
    time_end   = torch.cuda.Event(enable_timing = True)
    torch.cuda.synchronize()
    time_start.record()
    NB_ITER = 10
    for iterx in range(NB_ITER):
      # The network is applied "NB_ITER" times to calculate the average runtime (the output is disregarded)
      apply_network_forward_pass(
        sinogram                               = input_sinogram.to(device).float(),
        sos                                    = sos.to(device).float(),
        network                                = network,
        TRANSFORM_TST_NETWORK_OUTPUT           = p.TRANSFORM_TST_NETWORK_OUTPUT,
        DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
    torch.cuda.synchronize()
    time_end.record()
    torch.cuda.synchronize()
    time_inference_ms = time_start.elapsed_time(time_end)
    print('Average inference time: ' + str(int(time_inference_ms / NB_ITER)) + 'ms')

  # Gather data back to CPU
  output_image   = output_image.detach().cpu().numpy().squeeze()
  input_sinogram = input_sinogram.detach().cpu().numpy().squeeze()
  sos_value      = sos.detach().cpu().numpy().squeeze()

  # Scale data to original scale
  input_sinogram = input_sinogram * p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION
  output_image   = output_image * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION

  # Display
  showcase_demo_DeepMB_ifr(
    output_image,
    reference_image,
    input_sinogram,
    sos_value,
    sinogram_filename,
    g,
    flipud)
