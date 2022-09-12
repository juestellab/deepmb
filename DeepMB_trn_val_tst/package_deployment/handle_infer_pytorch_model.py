import os
import torch
from package_networks     import network_DeepMB
from .infer_pytorch_model import infer_pytorch_model


def handle_infer_pytorch_model(
  input_sinogram, input_sos, path_to_pytorch_model, pytorch_model_name, device, use_cuda, use_tracing, NB_REPETITIONS, p, g):

  # Instantiate the Network
  network = network_DeepMB.Net(
    torch.from_numpy(g.transducer_pixel_distances).to(device), g.F_SAMPLE, g.SAMPLES_OFFSET, device, p, g)
  network.float()
  if use_cuda:
    network.cuda(device=device)

  # Load the trained network and set it in "eval" mode
  network_checkpoint_path_and_filename = os.path.join(path_to_pytorch_model, pytorch_model_name)
  checkpoint = torch.load(network_checkpoint_path_and_filename, map_location=device)
  network.load_state_dict(checkpoint['network_state_dict'])
  network.eval()

  # Optional optimization that slightly reduces the inference time of the network
  if use_tracing:
    network = torch.jit.trace(network, (input_sinogram, input_sos))

  # Warm-up for timing
  output_image = infer_pytorch_model(input_sinogram, input_sos, network, device, p)

  # Synchronous timing for info (the previous call to "apply_network_forward_pass" was used as the warm-up procedure)
  time_start = torch.cuda.Event(enable_timing = True)
  time_end = torch.cuda.Event(enable_timing = True)
  torch.cuda.synchronize()
  time_start.record()
  for iterx in range(NB_REPETITIONS):
    infer_pytorch_model(input_sinogram, input_sos, network, device, p)
  torch.cuda.synchronize()
  time_end.record()
  torch.cuda.synchronize()
  time_inference_ms = time_start.elapsed_time(time_end)

  return output_image, time_inference_ms
