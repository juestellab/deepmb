import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_colorbar(im, axx):
  divider = make_axes_locatable(axx)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = plt.colorbar(im, cax=cax)
  return cbar


# EXAMPLE USE:
"""
import numpy as np
from nice_colorbar import nice_colorbar
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

axx = ax[0]
fig.sca(axx)
plt.title('a') # ---> The title should be defined ___before___ the call to "nice_colorbar"
im = axx.imshow(np.random.rand(100,100))
cbar = nice_colorbar(im, axx)
cbar.set_ticks([<...>])
cbar.set_ticklabels([<...>])
cbar.ax.tick_params(labelsize=<...>)

axx = ax[1]
fig.sca(axx)
plt.title('b') # ---> The title should be defined ___before___ the call to "nice_colorbar"
im = axx.imshow(np.random.rand(20,20))
cbar = nice_colorbar(im, axx)
cbar.set_ticks([<...>])
cbar.set_ticklabels([<...>])
cbar.ax.tick_params(labelsize=<...>)

plt.show()
"""
