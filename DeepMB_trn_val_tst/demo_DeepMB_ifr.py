import numpy as np
import os
import sys
import torch
from medpy.io    import load
from torchvision import transforms

from package_geometry.define_scanner_and_fov_geometry           import define_scanner_and_fov_geometry
from package_utils.environment_check                            import environment_check
from package_deployment.prepare_data_for_demo                   import prepare_data_for_demo
from package_deployment.handle_infer_pytorch_model              import handle_infer_pytorch_model
from package_deployment.display_deployment_results_pytorch_onnx import display_deployment_results_pytorch_onnx
from package_deployment.display_deployment_results_pytorch      import display_deployment_results_pytorch


# NOTE: Depending on your local environment and your own scikit configuration, you may encounter a conflict...
# ... preventing the final figure to be displayed: in that case, please uncomment the line below:
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':

  # ----------------------------------------------------------------
  # --- <Parameters> -----------------------------------------------
  # ----------------------------------------------------------------

  # Name of the sample (including extension, e.g., ".nii")
  # Please select one file that is provided on our GitHub:
  # "https://github.com/juestellab/deepmb/tree/binaries/in_vivo_data/sinograms"
  # NOTE: All the necessary files will be automatically downloaded by this demo script
  sample_filename = 'dmb_orange_p06_biceps_left_longitudinal_01_sosGt_1485_sosRec_1485_800nm.nii'

  # Speed of sound [m/s]
  # Please note that the optimal speed of sound is provided in the "sample_filename":
  # "dmb_orange_<participant number>_<anatomy>_<probe orientation>_<repetition>_...
  # ...<SoS determined for ground truth>_<SoS used for MB reconstruction>_<wavelength>.nii"
  # NOTE: Recommened values are [1475, 1480, 1485, 1490, 1495, 1500, 1505, 1510, 1515, 1520, 1525] m/s
  sos_value = 1485

  # Use a just-in-time compiled version of the Pytorch model that slightly reduces the inference time
  use_tracing = True

  # Use the ONNX (Open Neural Network Exchange) compiled version of the DeepMB model for notable speed-up
  # NOTE: This requires the installation of the Python packages "onnxruntime" and "onnxruntime-gpu"
  use_onnx = False

  # ----------------------------------------------------------------
  # --- </Parameters> ----------------------------------------------
  # ----------------------------------------------------------------

  # Import the 'get_parameters' module from the specified experiment
  parameters_path = os.path.join('package_deployment', 'stored_parameters_cd49')
  sys.path.insert(0, parameters_path)
  from get_parameters import get_parameters

  # Get the experiment parameters
  p = get_parameters()

  # Define the geometry corresponding to the imaging system and the field of view
  g = define_scanner_and_fov_geometry(p.SCANNER_ID)

  # Sanity test
  use_cuda, device, _ = environment_check(p.DETERMINISTIC, p.RANDOM_SEED, p.GPU_INDEX)

  # Data for the demo: create the local folder structure and download the in vivo samples as well as the trained models
  path_to_pytorch_model, path_to_onnx_model, pytorch_model_name, onnx_model_name, path_to_mb_images, path_to_sinograms = \
    prepare_data_for_demo()

  # Load the sinogram
  original_sinogram, _ = load(os.path.join(path_to_sinograms, sample_filename))

  # Load the target reference MB reconstruction
  target_reference_image, _ = load(os.path.join(path_to_mb_images, sample_filename))

  # Define the Tensor transformation that is applied to the sinograms
  transform_input = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION])])

  # Tensor-transform the data (sinogram as well as SoS value), cast to appropriate type, and load to GPU memory
  input_sinogram_gpu = transform_input(original_sinogram).unsqueeze(0).float().to(device)
  input_sos_gpu      = torch.tensor(float(sos_value)).unsqueeze(0).float().to(device)

  # The models will be called several times to calculate the average inference time
  NB_REPETITIONS = 25

  # Define and apply the "vanilla" PyTorch model
  output_ima_pytorch, time_inference_ms_pytorch = handle_infer_pytorch_model(
    input_sinogram_gpu, input_sos_gpu, path_to_pytorch_model, pytorch_model_name, device, use_cuda, use_tracing, NB_REPETITIONS, p, g)

  # Define and apply the compiled ONNX model
  if use_onnx:
    from package_deployment.handle_infer_onnx_model_with_iobinding_to_pytorch_tensor import handle_infer_onnx_model_with_iobinding_to_pytorch_tensor
    output_ima_onnx, time_inference_ms_onnx = handle_infer_onnx_model_with_iobinding_to_pytorch_tensor(
      input_sinogram_gpu, input_sos_gpu, path_to_onnx_model, onnx_model_name, NB_REPETITIONS, p, g)

  # Log the inference time
  time_per_frame_pytorch  = int(time_inference_ms_pytorch / NB_REPETITIONS)
  print('[PyTorch] \t Average inference time: ' + str(time_per_frame_pytorch) + 'ms')
  if use_onnx:
    time_per_frame_onnx = int(time_inference_ms_onnx / NB_REPETITIONS)
    print('[ONNX] \t Average inference time: ' + str(time_per_frame_onnx) + 'ms')

  # Display
  if use_onnx:
    display_deployment_results_pytorch_onnx(
      original_sinogram, sos_value, target_reference_image, output_ima_pytorch, output_ima_onnx, sample_filename, g,
      time_per_frame_pytorch, time_per_frame_onnx)
  else:
    display_deployment_results_pytorch(
      original_sinogram, sos_value, target_reference_image, output_ima_pytorch, sample_filename, g, time_per_frame_pytorch)
