from medpy.io import load, save
import os


def load_nifti(file_path, file_name):
  image_data, image_header = load(os.path.join(file_path, file_name + '.nii'))
  return image_data, image_header


def save_nifti(data, file_path, file_name):
  save(data, os.path.join(file_path, file_name + '.nii'))
