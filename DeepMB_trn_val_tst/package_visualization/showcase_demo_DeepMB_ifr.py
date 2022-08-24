import numpy as np
import matplotlib.pyplot as plt
from .nice_colorbar import nice_colorbar


def showcase_demo_DeepMB_ifr(ima_deepmb, ima_mb, sin, sos_value, name, g, flipud):

  if flipud: # (in vivo output images are upside-down)
    ima_deepmb = np.flipud(ima_deepmb)

  if ima_mb is not None:
    fig, ax = plt.subplots(1, 3)
    MAX_CLIM = 0.6 * np.max(ima_mb)
  else:
    fig, ax = plt.subplots(1, 2)
    MAX_CLIM = 0.6 * np.max(ima_deepmb)

  MILLIMETERS = 1000

  axx = ax[0]
  fig.sca(axx)
  im = axx.imshow(np.transpose(sin), cmap='viridis')
  axx.set_title('Input sinogram')
  plt.xlabel('Time samples')
  plt.ylabel('Channels')
  nice_colorbar(im, axx)

  axx = ax[1]
  fig.sca(axx)
  im = axx.imshow(ima_deepmb, cmap = 'bone', clim=[0, MAX_CLIM])
  axx.set_title(
    'Output image (DeepMB) \n' +
    name +
    '\n SoS (m/s) = ' + str(int(sos_value)) +
    '; FoV (pixels) = ' + str(g.NB_PIXELS_X) + r'$\times$' + str(g.NB_PIXELS_Z) +
    r'; FoV (mm$^2$) = ' + str(MILLIMETERS * g.IMA_SIZE_X) + r'$\times$' + str(MILLIMETERS * g.IMA_SIZE_Z))
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  if ima_mb is not None:
    axx = ax[2]
    fig.sca(axx)
    im = axx.imshow(np.flipud(ima_mb), cmap = 'bone', clim=[0, MAX_CLIM])
    axx.set_title('Reference image (model-based)')
    plt.xticks([])
    plt.yticks([])
    nice_colorbar(im, axx)

  plt.show()
