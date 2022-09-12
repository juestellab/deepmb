import torch
from package_networks.apply_network_forward_pass import apply_network_forward_pass


def infer_pytorch_model(input_sinogram, input_sos, network, device, p):

  # Disable gradient calculation to accelerate inference
  with torch.no_grad():

    # Apply the network forward pass
    output_image = apply_network_forward_pass(
      sinogram                               = input_sinogram.to(device).float(),
      sos                                    = input_sos.to(device).float(),
      network                                = network,
      TRANSFORM_TST_NETWORK_OUTPUT           = p.TRANSFORM_TST_NETWORK_OUTPUT,
      DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

  # Gather the variables back to CPU
  output_image = output_image.detach().cpu().numpy().squeeze()

  # Scale data to original scale
  output_image = output_image * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION

  return output_image
