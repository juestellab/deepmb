import torch
from .create_ort_session                                import create_ort_session
from .infer_onnx_model_with_iobinding_to_pytorch_tensor import infer_onnx_model_with_iobinding_to_pytorch_tensor
from .gather_ort_output                                 import gather_ort_output


def handle_infer_onnx_model_with_iobinding_to_pytorch_tensor(
  input_sinogram, input_sos, path_to_onnx_model, onnx_model_name, NB_REPETITIONS, p, g):

  # Create the ONNX session
  ort_session = create_ort_session(path_to_onnx_model, onnx_model_name)

  # Warm-up for timing
  output_image = infer_onnx_model_with_iobinding_to_pytorch_tensor(input_sinogram, input_sos, ort_session, g)
  output_image = gather_ort_output(output_image, p, ONNX_OUTPUT_ALREADY_AS_NUMPY = False)

  # Synchronous timing for info (the previous call to "apply_network_forward_pass" was used as the warm-up procedure)
  time_start = torch.cuda.Event(enable_timing = True)
  time_end = torch.cuda.Event(enable_timing = True)
  torch.cuda.synchronize()
  time_start.record()
  for iterx in range(NB_REPETITIONS):
    infer_onnx_model_with_iobinding_to_pytorch_tensor(input_sinogram, input_sos, ort_session, g)
  torch.cuda.synchronize()
  time_end.record()
  torch.cuda.synchronize()
  time_inference_ms = time_start.elapsed_time(time_end)

  return output_image, time_inference_ms
