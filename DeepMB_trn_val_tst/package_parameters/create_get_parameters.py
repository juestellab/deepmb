import os
from shutil import copyfile


def create_get_parameters(stored_parameters_output_dir):

  # Copy the Git-ignored file "set_parameters.py" into the Git-tracked file "get_parameters.py"
  copyfile(
    os.path.join('package_parameters', 'set_parameters.py'),
    os.path.join('package_parameters', 'get_parameters.py'))

  # Within "get_parameters.py", modify the function name from "set_parameters(...)" to "get_parameters(...)"
  fid = open(os.path.join('package_parameters', 'get_parameters.py'), 'rt')
  data = fid.read()
  data = data.replace('set_parameters()', 'get_parameters()')
  fid.close()
  fid = open(os.path.join('package_parameters', 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # Make a Git-ignored copy of the parameter files needed to re-run the experiment, and store them next to the results
  copyfile(
    os.path.join('package_parameters', 'get_parameters.py'),
    os.path.join(stored_parameters_output_dir, 'get_parameters.py'))
  copyfile(
    os.path.join('package_parameters', 'parameters.py'),
    os.path.join(stored_parameters_output_dir, 'parameters.py'))
  copyfile(
    os.path.join('package_parameters', '__init__.py'),
    os.path.join(stored_parameters_output_dir, '__init__.py'))
