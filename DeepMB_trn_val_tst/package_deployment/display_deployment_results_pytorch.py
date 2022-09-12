import numpy as np
import matplotlib.pyplot as plt
from package_visualization.nice_colorbar import nice_colorbar


def display_deployment_results_pytorch(
  input_sinogram, input_sos, target_reference_image, output_pytorch_image, sample_filename, g, time_per_frame_pytorch):

  # Layout: [Sinogram, Reference MB image, Output DeepMB image, Difference MB-DeepMB]
  fig, ax = plt.subplots(1, 4)

  # Suptitle
  plt.suptitle(
    sample_filename +
    '\n SoS (m/s) = ' + str(input_sos) +
    '; FoV (pixels) = ' + str(g.NB_PIXELS_X) + r'$\times$' + str(g.NB_PIXELS_Z) +
    r'; FoV (mm$^2$) = ' + str(1000 * g.IMA_SIZE_X) + r'$\times$' + str(1000 * g.IMA_SIZE_Z))

  # Threshold the high intensity values to reduce the total image dynamic and improve visualization
  VMAX = 0.6 * np.max(target_reference_image)

  axx = ax[0]
  fig.sca(axx)
  im = axx.imshow(np.transpose(input_sinogram), cmap='viridis')
  axx.set_title('Input sinogram')
  plt.xlabel('Time samples')
  plt.ylabel('Channels')
  nice_colorbar(im, axx)

  axx = ax[1]
  fig.sca(axx)
  im = axx.imshow(np.flipud(target_reference_image), cmap = 'bone', clim=[0, VMAX])
  axx.set_title('Reference MB image')
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  axx = ax[2]
  fig.sca(axx)
  im = axx.imshow(np.flipud(output_pytorch_image), cmap = 'bone', clim=[0, VMAX])
  axx.set_title('DeepMB image (native Pytorch model, ' + str(time_per_frame_pytorch) + 'ms)')
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  axx = ax[3]
  fig.sca(axx)
  im = axx.imshow(abs(np.flipud(target_reference_image - output_pytorch_image)), cmap = 'viridis')
  axx.set_title('Abs. difference |MB - DeepMB_PyTorch|')
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  plt.show()
