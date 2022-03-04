from .custom_transform import custom_transform


def apply_network_forward_pass(sinogram, sos, network, **kwargs):
  # Apply network forward pass
  ima_net = network(sinogram, sos)
  # Verify if the network output needs to be transformed (cf the DeepMB paper)
  if 'TRANSFORM_TST_NETWORK_OUTPUT' in kwargs and 'DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION' in kwargs:
    ima_net = custom_transform(
      ima_net, kwargs['TRANSFORM_TST_NETWORK_OUTPUT'], kwargs['DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION'])
  return ima_net
