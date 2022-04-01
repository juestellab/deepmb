import numpy as np
import matplotlib.pyplot as plt
from .nice_colorbar import nice_colorbar


def showcase_demo_DeepMB_ifr(ima, sin, sos, name, g):

  fig, ax = plt.subplots(1, 2)

  axx = ax[0]
  fig.sca(axx)
  im = axx.imshow(np.transpose(sin), cmap='viridis')
  axx.set_title('Input sinogram')
  plt.xlabel('Time samples')
  plt.ylabel('Channels')
  nice_colorbar(im, axx)

  MILLIMETERS = 1000
  MAX_CLIM = 0.6

  axx = ax[1]
  fig.sca(axx)
  im = axx.imshow(ima, cmap = 'bone', clim=[0, MAX_CLIM * np.max(ima)])
  axx.set_title(
    'Output image \n' +
    name +
    '\n SoS (m/s) = ' + str(sos) +
    '; FoV (pixels) = ' + str(g.NB_PIXELS_X) + r'$\times$' + str(g.NB_PIXELS_Z) +
    r'; FoV (mm$^2$) = ' + str(MILLIMETERS * g.IMA_SIZE_X) + r'$\times$' + str(MILLIMETERS * g.IMA_SIZE_Z))
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  plt.show()
