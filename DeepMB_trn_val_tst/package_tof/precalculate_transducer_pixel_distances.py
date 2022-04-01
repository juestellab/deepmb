import numpy as np


def precalculate_transducer_pixel_distances(
  sensor_x, sensor_z, ima_size_x, ima_size_z, nb_pixels_x, nb_pixels_z, nb_elements):

  pos_x = np.linspace(-0.5 * ima_size_x, 0.5 * ima_size_x, nb_pixels_x)
  pos_z = np.linspace(-0.5 * ima_size_z, 0.5 * ima_size_z, nb_pixels_z)
  grid_x, grid_z = np.meshgrid(pos_x, pos_z)
  transducer_pixel_distances = np.zeros((nb_pixels_z, nb_pixels_x, nb_elements), dtype=np.double)

  for id_el in range(nb_elements):
    transducer_pixel_distances[:, :, id_el] = ((grid_x - sensor_x[id_el])**2 + (grid_z - sensor_z[id_el])**2)**0.5

  return transducer_pixel_distances
