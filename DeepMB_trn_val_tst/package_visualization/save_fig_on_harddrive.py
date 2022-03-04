import os


def save_fig_on_harddrive(fig, print_dir, file_name, **kwargs):

  if 'trn_val_tst' in kwargs and 'id_epoch' in kwargs:
    # These fields are useful during training to monitor the progress
    prefix = kwargs['trn_val_tst'] + '_' + str(kwargs['id_epoch']) + '_'
  else:
    # These fields are not required anymore during deployment (when called by "batch_DeepMB_ifr.py")
    prefix = ''

  fig_name = prefix + str(file_name)
  fig.savefig(os.path.join(print_dir, fig_name + '.png'), bbox_inches = 'tight')
