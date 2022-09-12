from package_networks.custom_transform import custom_transform


def gather_ort_output(ort_output, p, ONNX_OUTPUT_ALREADY_AS_NUMPY = False):

  if ONNX_OUTPUT_ALREADY_AS_NUMPY:

    # There is only one output (the reconstructed image)
    ima_output = ort_output[0]

    # This corresponds to the post-processing operation of DeepMB
    ima_output = ima_output ** 2
    ima_output = ima_output * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION
    ima_output = ima_output.squeeze()

  else:

    # Apply the custom scaling transformation to return to the original intensity range
    ima_output = custom_transform(ort_output, p.TRANSFORM_TST_NETWORK_OUTPUT, p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

    # Gather data back to CPU and un-Tensor it
    ima_output = ima_output.detach().cpu().numpy().squeeze()

    # Scale to original size
    ima_output = ima_output * p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION

  return ima_output
