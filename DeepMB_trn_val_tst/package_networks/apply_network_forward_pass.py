from .custom_transform import custom_transform


def apply_network_forward_pass(sinogram, sos, network, **kwargs):

  # Apply network forward pass
  ima_net = network(sinogram, sos)

  # Transform back the network output if necessary (cf the DeepMB paper)
  if 'TRANSFORM_TST_NETWORK_OUTPUT' in kwargs and 'DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION' in kwargs:
    ima_net = custom_transform(
      ima_net, kwargs['TRANSFORM_TST_NETWORK_OUTPUT'], kwargs['DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION'])

  return ima_net
