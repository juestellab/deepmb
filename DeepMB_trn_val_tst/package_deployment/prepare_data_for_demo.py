import os
import hashlib
import requests
from package_utils.create_folder_if_not_exist import create_folder_if_not_exist


# ----------------------------------------------------------------
def get_readable_hash_from_file(file_path, file_name_and_extension):
  with open(os.path.join(file_path, file_name_and_extension), 'rb') as fid:
    return hashlib.sha256(fid.read()).hexdigest()


# ----------------------------------------------------------------
def download_from_github(download_link, destination_path, destination_filename_and_extension):
    # Check whether the file is already present locally
    if not os.path.exists(os.path.join(destination_path, destination_filename_and_extension)):
      # Ensure that the destination folder exists locally
      create_folder_if_not_exist(destination_path)
      # Download the file
      print('Downloading: ' + str(download_link))
      r = requests.get(download_link)
      with open(os.path.join(destination_path, destination_filename_and_extension), 'wb') as fid:
        fid.write(r.content)


# ----------------------------------------------------------------
def check_model_correctness_otherwise_raise_error(path_to_model, model_name, reference_sha256):
  model_hash = get_readable_hash_from_file(path_to_model, model_name)
  if model_hash != reference_sha256:
    raise ValueError('The hash of the model ' + str(model_name) + ' is incorrect')


# ----------------------------------------------------------------
def define_local_folder_structure(demo_root):
  path_to_models        = os.path.join(demo_root, 'trained_models')
  path_to_pytorch_model = os.path.join(path_to_models, 'DeepMB_trained_model_cd49_PyTorch')
  path_to_onnx_model    = os.path.join(path_to_models, 'DeepMB_trained_model_cd49_ONNX')
  path_to_in_vivo_data  = os.path.join(demo_root, 'in_vivo_data')
  path_to_mb_images     = os.path.join(path_to_in_vivo_data, 'mb_images')
  path_to_sinograms     = os.path.join(path_to_in_vivo_data, 'sinograms')
  return path_to_pytorch_model, path_to_onnx_model, path_to_mb_images, path_to_sinograms


# ----------------------------------------------------------------
def check_in_vivo_data_availability_otherwise_download(url_prefix, filenames, destination_path):
  for idx in range(len(filenames)):
    download_from_github(url_prefix + filenames[idx], destination_path, filenames[idx])


# ----------------------------------------------------------------
def prepare_data_for_demo():

  # NOTE: This script assumes that several files are present on GitHub, these hard-coded parts are taggged with "(GZ 2022-09-08)"

  demo_root                   = 'DeepMB_deployed_demo' # (GZ 2022-09-08)
  pytorch_model_name          = 'model_min_val_loss.pt' # (GZ 2022-09-08)
  onnx_model_name             = 'model_min_val_loss.onnx' # (GZ 2022-09-08)
  github_prefix               = 'https://github.com/juestellab/deepmb/raw/binaries/' # (GZ 2022-09-08)
  pytorch_model_download_link = github_prefix + 'DeepMB_trained_model_cd49_PyTorch/model_min_val_loss.pt' # (GZ 2022-09-08)
  onnx_model_download_link    = github_prefix + 'DeepMB_trained_model_cd49_ONNX/model_min_val_loss.onnx' # (GZ 2022-09-08)
  pytorch_model_sha256        = 'ecf826911a52798118966a3496df1eac6ce2926fa52e265072b8dc2e516dcd13' # (GZ 2022-09-08)
  onnx_model_sha256           = '334d74915b883f2d875672aec007f3e05136d4296594583c0ec58ef0ac834717' # (GZ 2022-09-08)
  mb_images_prefix             = github_prefix + 'in_vivo_data/mb_images/' # (GZ 2022-09-08)
  sinograms_prefix             = github_prefix + 'in_vivo_data/sinograms/' # (GZ 2022-09-08)
  filenames                   = []

  filenames.append('dmb_orange_p01_neck_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_neck_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_thyroid_left_longitudinal_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_biceps_right_transversal_01_sosGt_1485_sosRec_1485_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_carotid_right_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_calf_right_longitudinal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_carotid_left_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_ulnar_left_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_biceps_right_longitudinal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_calf_left_longitudinal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_calf_right_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_biceps_left_transversal_01_sosGt_1485_sosRec_1485_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_calf_left_longitudinal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_thyroid_left_longitudinal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_thyroid_right_longitudinal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_thyroid_right_transversal_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_colon_left_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_carotid_right_longitudinal_01_sosGt_1515_sosRec_1515_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_calf_right_longitudinal_01_sosGt_1520_sosRec_1520_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_biceps_left_transversal_02_sosGt_1480_sosRec_1480_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_carotid_right_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_calf_left_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_biceps_right_transversal_01_sosGt_1490_sosRec_1490_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_ulnar_right_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_biceps_right_longitudinal_01_sosGt_1475_sosRec_1475_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_biceps_left_transversal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_thyroid_right_longitudinal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_ulnar_right_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_calf_right_transversal_01_sosGt_1490_sosRec_1490_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_thyroid_left_transversal_01_sosGt_1515_sosRec_1515_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_carotid_left_longitudinal_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_colon_right_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_biceps_left_longitudinal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_biceps_left_longitudinal_01_sosGt_1485_sosRec_1485_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_ulnar_left_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_carotid_right_longitudinal_01_sosGt_1515_sosRec_1515_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_colon_right_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_thyroid_left_transversal_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_colon_left_01_sosGt_1510_sosRec_1510_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_thyroid_right_transversal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p06_carotid_left_transversal_01_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_carotid_left_longitudinal_01_sosGt_1500_sosRec_1500_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_carotid_right_transversal_02_sosGt_1505_sosRec_1505_800nm.nii') # (GZ 2022-09-08)
  filenames.append('dmb_orange_p01_calf_left_transversal_01_sosGt_1495_sosRec_1495_800nm.nii') # (GZ 2022-09-08)

  # Define the folder structure that is needed to deploy the demo inference
  path_to_pytorch_model, path_to_onnx_model, path_to_mb_images, path_to_sinograms = \
    define_local_folder_structure(demo_root)

  # Verify that the models are available, otherwise download them
  download_from_github(pytorch_model_download_link, path_to_pytorch_model, pytorch_model_name)
  download_from_github(onnx_model_download_link, path_to_onnx_model, onnx_model_name)

  # Verify that the models are correct
  check_model_correctness_otherwise_raise_error(path_to_pytorch_model, pytorch_model_name, pytorch_model_sha256)
  check_model_correctness_otherwise_raise_error(path_to_onnx_model, onnx_model_name, onnx_model_sha256)

  # Verify that the data is available, otherwise download it
  check_in_vivo_data_availability_otherwise_download(mb_images_prefix, filenames, path_to_mb_images)
  check_in_vivo_data_availability_otherwise_download(sinograms_prefix, filenames, path_to_sinograms)

  return \
    path_to_pytorch_model, path_to_onnx_model, pytorch_model_name, onnx_model_name, path_to_mb_images, path_to_sinograms
