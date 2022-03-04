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

  axx = ax[1]
  fig.sca(axx)
  im = axx.imshow(ima, cmap = 'bone', clim=[0, 0.6*np.max(ima)])
  axx.set_title(
    'Output image \n' +
    name +
    '\n SoS (m/s) = ' + str(sos) +
    '; FoV (pixels) = ' + str(g.NB_PIXELS_X) + r'$\times$' + str(g.NB_PIXELS_Y) +
    r'; FoV (mm$^2$) = ' + str(1000*g.IMA_SIZE_X) + r'$\times$' + str(1000*g.IMA_SIZE_Y))
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  plt.show()
