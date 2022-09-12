import os
import onnxruntime as ort


def create_ort_session(path_to_onnx_model, onnx_model_name):

  # Session options
  session_options                  = ort.SessionOptions()
  session_options.execution_mode   = ort.ExecutionMode.ORT_PARALLEL
  session_options.enable_profiling = False

  ort_session = ort.InferenceSession(
    os.path.join(path_to_onnx_model, onnx_model_name),
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

  # Verify that the session runs on the GPU
  # NOTE: It is requested to "pip3 install onnxruntime-gpu", because the vanilla "onnxruntime" only runs on CPU
  print('The ONNX RunTime session is running on: ' + str(ort.get_device()))

  return ort_session
