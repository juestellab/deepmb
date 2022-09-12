import numpy as np
import torch


def infer_onnx_model_with_iobinding_to_pytorch_tensor(input_sinogram, input_sos, ort_session, g):

  # Input-Output binding to utilize "data on device" (the field names correspond to the ones used during "export(.)")
  io_binding = ort_session.io_binding()

  # Ensure the tensors are contiguous in memory
  input_sinogram_tensor = input_sinogram.contiguous()
  input_sos_tensor      = input_sos.contiguous()

  # Define the device
  # TODO: Ideally, the variables "device_type", "device_id", and "device" should be set via input parameters
  device_type = 'cuda'
  device_id   = 0

  # Specify the two model inputs
  io_binding.bind_input(
    name         = 'input_sinogram',
    device_type  = device_type,
    device_id    = device_id,
    element_type = np.float32,
    shape        = tuple(input_sinogram_tensor.shape),
    buffer_ptr   = input_sinogram_tensor.data_ptr())

  io_binding.bind_input(
    name         = 'input_sos',
    device_type  = device_type,
    device_id    = device_id,
    element_type = np.float32,
    shape        = tuple(input_sos_tensor.shape),
    buffer_ptr   = input_sos_tensor.data_ptr())

  # Allocate the PyTorch tensor for the model output
  output_shape = (1, 1, g.NB_PIXELS_Z, g.NB_PIXELS_X)
  output_tensor = torch.empty(output_shape, dtype=torch.float32, device='cuda:0').contiguous()
  io_binding.bind_output(
    name         = 'output_image',
    device_type  = device_type,
    device_id    = device_id,
    element_type = np.float32,
    shape        = tuple(output_tensor.shape),
    buffer_ptr   = output_tensor.data_ptr())

  # Compute ONNX Runtime output prediction
  ort_session.run_with_iobinding(io_binding)

  return output_tensor
