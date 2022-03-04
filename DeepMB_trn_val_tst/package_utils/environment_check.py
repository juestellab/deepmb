import struct
import scipy
import sys
import torch
import torchvision


def environment_check(deterministic, random_seed, gpu_index):

  # Determine whether a GPU is available
  gpu_available = torch.cuda.is_available()
  device = torch.device('cuda:%i' % gpu_index if gpu_available else 'cpu')

  # Seed the randomness for fully reproducible results
  if deterministic:
    # Training may be slower to run due to less optimization shortcuts
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    num_workers = 0
    torch.manual_seed(random_seed)
    if gpu_available:
      torch.cuda.manual_seed(random_seed)
      torch.cuda.manual_seed_all(random_seed)
  else:
    # Random, not reproducible, but slightly faster to run
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    num_workers = 4

  torch.backends.cudnn.enabled = True

  # Console log
  print('Python version: ' + sys.version)
  print('SciPy version: ' + str(scipy.__version__))
  print('PyTorch version: ', torch.__version__)
  print('torchvision version: ', torchvision.__version__)
  print('Bits: ' + str(struct.calcsize('P') * 8))  # 32 or 64 bits
  print('CUDA version: ', torch.version.cuda)
  print('Pytorch CUDA availability: ' + str(gpu_available))
  print('Device: ' + str(device))
  print('----------------------------------------------------------------')

  return gpu_available, device, num_workers
