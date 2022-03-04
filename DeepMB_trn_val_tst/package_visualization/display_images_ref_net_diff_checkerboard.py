import numpy as np
import matplotlib.pyplot as plt
from .nice_colorbar import nice_colorbar


def display_images_ref_net_diff_checkerboard(ima_ref, ima_net, name, id_epoch, trn_val_tst, flipud):

  # In vivo images are upside-down
  if flipud:
    ima_ref = np.flipud(ima_ref)
    ima_net = np.flipud(ima_net)

  # Signed difference between the reference image and the network image
  ima_diff = ima_ref - ima_net

  # Determine if "ima_net" is too different from "ima_ref", in which case "clim" is not applied to better view "ima_net"
  # It is expected that at least "THRESHOLD_CLIM" pixels of "ima_net" are within the bounds of "ima_ref" magnitude
  THRESHOLD_CLIM = 0.75
  values_within_clim = (ima_net <= np.max(ima_ref)).astype(int) * (ima_net >= np.min(ima_ref)).astype(int)
  nb_values_within_clim = np.sum(values_within_clim) / values_within_clim.size
  CLIM = nb_values_within_clim > THRESHOLD_CLIM

  # Zoomed ROI (these values correspond to 128*128 pixels)
  ROI_X = [127, 255]
  ROI_Y = [154, 281]

  # Checkerboard (full: 8*52=416) (zoom: 8*16=128)
  NB_SQUARES = 8                        # Number of square patches alternating between the reference image and the network image
  DIM_F      = 416                      # Full, image size
  DIM_Z      = 128                      # Zoom, image size
  SQR_F      = 52                       # Full, square size
  SQR_Z      = 16                       # Zoom, square size
  checker_f  = np.zeros((DIM_F, DIM_F)) # Full, checkerboard image
  checker_z  = np.zeros((DIM_Z, DIM_Z)) # Zoom, checkerboard image
  for x in range(0, NB_SQUARES, 2):
    for y in range(0, NB_SQUARES, 2):
      # Full image
      checker_f[(x+0)*SQR_F:(x+1)*SQR_F, (y+0)*SQR_F:(y+1)*SQR_F] = \
        ima_ref[(x+0)*SQR_F:(x+1)*SQR_F, (y+0)*SQR_F:(y+1)*SQR_F]
      checker_f[(x+1)*SQR_F:(x+2)*SQR_F, (y+1)*SQR_F:(y+2)*SQR_F] = \
        ima_ref[(x+1)*SQR_F:(x+2)*SQR_F, (y+1)*SQR_F:(y+2)*SQR_F]
      checker_f[(x+1)*SQR_F:(x+2)*SQR_F, (y+0)*SQR_F:(y+1)*SQR_F] = \
        ima_net[(x+1)*SQR_F:(x+2)*SQR_F, (y+0)*SQR_F:(y+1)*SQR_F]
      checker_f[(x+0)*SQR_F:(x+1)*SQR_F, (y+1)*SQR_F:(y+2)*SQR_F] = \
        ima_net[(x+0)*SQR_F:(x+1)*SQR_F, (y+1)*SQR_F:(y+2)*SQR_F]
      # Zoom image
      checker_z[(x+0)*SQR_Z:(x+1)*SQR_Z, (y+0)*SQR_Z:(y+1)*SQR_Z] = \
        ima_ref[ROI_Y[0]+(x+0)*SQR_Z:ROI_Y[0]+(x+1)*SQR_Z, ROI_X[0]+(y+0)*SQR_Z:ROI_X[0]+(y+1)*SQR_Z]
      checker_z[(x+1)*SQR_Z:(x+2)*SQR_Z, (y+1)*SQR_Z:(y+2)*SQR_Z] = \
        ima_ref[ROI_Y[0]+(x+1)*SQR_Z:ROI_Y[0]+(x+2)*SQR_Z, ROI_X[0]+(y+1)*SQR_Z:ROI_X[0]+(y+2)*SQR_Z]
      checker_z[(x+1)*SQR_Z:(x+2)*SQR_Z, (y+0)*SQR_Z:(y+1)*SQR_Z] = \
        ima_net[ROI_Y[0]+(x+1)*SQR_Z:ROI_Y[0]+(x+2)*SQR_Z, ROI_X[0]+(y+0)*SQR_Z:ROI_X[0]+(y+1)*SQR_Z]
      checker_z[(x+0)*SQR_Z:(x+1)*SQR_Z, (y+1)*SQR_Z:(y+2)*SQR_Z] = \
        ima_net[ROI_Y[0]+(x+0)*SQR_Z:ROI_Y[0]+(x+1)*SQR_Z, ROI_X[0]+(y+1)*SQR_Z:ROI_X[0]+(y+2)*SQR_Z]

  # Figure (will later be either printed on the hard drive, or written in Tensorboard)
  fig, ax = plt.subplots(2, 4)
  fig.set_size_inches(w=30, h=16)

  ### Full-size

  # MB
  axx = ax[0, 0]
  fig.sca(axx)
  im = axx.imshow(ima_ref, cmap='bone')
  axx.set_title('MB')
  plt.xticks([])
  plt.yticks([])
  plt.plot(ROI_X[0], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[0], ROI_Y[1], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[1], '+', color='pink')
  clim = im.properties()['clim']
  nice_colorbar(im, axx)

  # DeepMB
  axx = ax[0, 1]
  fig.sca(axx)
  if CLIM:
    im = axx.imshow(ima_net, clim=clim, cmap='bone')
  else:
    im = axx.imshow(ima_net, cmap='cool')
  axx.set_title('Epoch: ' + str(id_epoch) + ' (' + trn_val_tst + ')\n[' + str(name) + ']\nDeepMB')
  plt.xticks([])
  plt.yticks([])
  plt.plot(ROI_X[0], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[0], ROI_Y[1], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[1], '+', color='pink')
  nice_colorbar(im, axx)

  # MB - DeepMB
  axx = ax[0, 2]
  fig.sca(axx)
  im = axx.imshow(ima_diff, cmap='viridis')
  axx.set_title('MB - DeepMB')
  plt.xticks([])
  plt.yticks([])
  plt.plot(ROI_X[0], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[0], ROI_Y[1], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[1], '+', color='pink')
  nice_colorbar(im, axx)

  # Checkerboard(MB, DeepMB)
  axx = ax[0, 3]
  fig.sca(axx)
  im = axx.imshow(checker_f, cmap='bone')
  axx.set_title('8x8 checkerboard view')
  plt.xticks(range(0, DIM_F, SQR_F))
  plt.yticks(range(0, DIM_F, SQR_F))
  axx.tick_params(labelbottom=False)
  axx.tick_params(labelleft=False)
  plt.plot(ROI_X[0], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[0], ROI_Y[1], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[0], '+', color='pink')
  plt.plot(ROI_X[1], ROI_Y[1], '+', color='pink')
  nice_colorbar(im, axx)

  ### ROI-cropped

  # MB
  axx = ax[1, 0]
  fig.sca(axx)
  im = axx.imshow(ima_ref[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]], cmap='bone')
  axx.set_title('MB')
  plt.xticks([])
  plt.yticks([])
  clim = im.properties()['clim']
  nice_colorbar(im, axx)

  # DeepMB
  axx = ax[1, 1]
  fig.sca(axx)
  if CLIM:
    im = axx.imshow(ima_net[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]], clim=clim, cmap='bone')
  else:
    im = axx.imshow(ima_net[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]], cmap='cool')
  axx.set_title('DeepMB')
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  # MB - DeepMB
  axx = ax[1, 2]
  fig.sca(axx)
  im = axx.imshow(ima_diff[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]], cmap='viridis')
  axx.set_title('MB - DeepMB')
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  # Checkerboard(MB, DeepMB)
  axx = ax[1, 3]
  fig.sca(axx)
  im = axx.imshow(checker_z, cmap='bone')
  axx.set_title('8x8 checkerboard view')
  plt.xticks(range(0, DIM_Z, SQR_Z))
  plt.yticks(range(0, DIM_Z, SQR_Z))
  axx.tick_params(labelbottom=False)
  axx.tick_params(labelleft=False)
  nice_colorbar(im, axx)

  return fig
