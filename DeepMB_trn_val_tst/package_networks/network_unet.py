import numpy as np
import torch
import torch.nn as nn


"""
THIS FILE WAS ADAPTED FROM:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

# ----------------------------------------------------------------
class Unet(nn.Module):

  def __init__(
    self, input_nc,
    output_nc,
    n_layers = 2,
    ngf = 64,
    norm_layer = nn.BatchNorm2d,
    kernel_size = (3,3),
    padding = (1,1),
    activation = nn.LeakyReLU(0.2, True),
    use_bias = True):
    """
    Parameters:
      input_nc (int) -- the number of channels in input images
      output_nc (int) -- the number of channels in output images
      n_layers (int) -- number of layers of the Unet, must be at least 2
      ngf (int) -- the number of filters in the last (outermost) conv layer
      norm_layer -- normalization layer
      use_bias -- flag if convolutional layers should include bias terms

    We construct the U-Net from the innermost layer to the outermost layer.
    It is a recursive process.
    """
    super(Unet, self).__init__()

    # Add innermost layer
    unet_block = UnetSkipConnectionBlock(
      outer_nc=ngf * 2**(n_layers-2),
      inner_nc=ngf * 2**(n_layers-1),
      kernel_size=kernel_size,
      padding=padding,
      activation=activation,
      input_nc=None,
      submodule=None,
      norm_layer=norm_layer,
      innermost=True,
      use_bias=use_bias)

    # Add intermediate layers
    for i in range(n_layers-2, 0, -1):
      unet_block = UnetSkipConnectionBlock(
        outer_nc=ngf * 2**(i-1),
        inner_nc=ngf * 2**i,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        input_nc=None,
        submodule=unet_block,
        norm_layer=norm_layer,
        use_bias=use_bias)

    # Add outermost layer
    unet_block = UnetSkipConnectionBlock(
      outer_nc=output_nc,
      inner_nc=ngf,
      kernel_size=kernel_size,
      padding=padding,
      activation=activation,
      input_nc=input_nc,
      submodule=unet_block,
      outermost=True,
      norm_layer=norm_layer,
      use_bias=use_bias)

    self.model = unet_block


  def forward(self, input_data):
    return self.model(input_data)


# ----------------------------------------------------------------
class UnetSkipConnectionBlock(nn.Module):
  """
  Defines the Unet submodule with skip connection.
  X -------------------identity-------------------------------------------------------------------(+)--> output
  |max-pool -- conv+norm+Relu -- conv+norm+Relu -- |submodule| -- conv+norm+Relu -- conv+norm+Relu -- upsample|
  """

  def __init__(
    self,
    outer_nc,
    inner_nc,
    kernel_size,
    padding,
    activation,
    input_nc = None,
    submodule = None,
    outermost = False,
    innermost = False,
    norm_layer = nn.BatchNorm2d,
    use_bias = False):
    """
    Construct a Unet submodule with skip connections.
    Parameters:
      outer_nc (int) -- the number of filters in the outer conv layer
      inner_nc (int) -- the number of filters in the inner conv layer
      input_nc (int) -- the number of channels in input images/features
      submodule (UnetSkipConnectionBlock) -- previously defined submodules
      outermost (bool) -- if this module is the outermost module
      innermost (bool) -- if this module is the innermost module
      norm_layer -- normalization layer
      user_dropout (bool) -- if use dropout layers.
    """

    super(UnetSkipConnectionBlock, self).__init__()
    self.outermost = outermost
    if input_nc is None:
      input_nc = outer_nc

    input_to_inner_conv = \
      nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate')
    outer_to_inner_conv = \
      nn.Conv2d(outer_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate')
    inner_to_inner_conv_1 = \
      nn.Conv2d(inner_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate')
    inner_to_inner_conv_2 = \
      nn.Conv2d(inner_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate')
    three_inner_to_inner_conv = \
      nn.Conv2d(3*inner_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias, padding_mode='replicate')
    inner_to_outer_conv = \
      nn.Conv2d(inner_nc, outer_nc, kernel_size=1, stride=1, padding=0, bias=use_bias, padding_mode='replicate')

    maxpool = nn.MaxPool2d(kernel_size=2)
    upsample = nn.Upsample(scale_factor=2)

    if outermost:
      down = [input_to_inner_conv, norm_layer(inner_nc), activation, inner_to_inner_conv_1, norm_layer(inner_nc), activation]
      up = [three_inner_to_inner_conv, norm_layer(inner_nc), activation, inner_to_inner_conv_2, norm_layer(inner_nc), activation, inner_to_outer_conv]
      model = down + [submodule] + up

    elif innermost:
      down = [maxpool, outer_to_inner_conv, norm_layer(inner_nc), activation, inner_to_inner_conv_1, norm_layer(inner_nc), activation]
      up = [upsample]
      model = down + up

    else:
      down = [maxpool, outer_to_inner_conv, norm_layer(inner_nc), activation, inner_to_inner_conv_1, norm_layer(inner_nc), activation]
      up = [three_inner_to_inner_conv, norm_layer(inner_nc), activation, inner_to_inner_conv_2, norm_layer(inner_nc), activation, upsample]
      model = down + [submodule] + up

    self.model = nn.Sequential(*model)


  def forward(self, x):
    if self.outermost:
      res = self.model(x)
      return res
    else:
      # Add skip connections
      res0 = self.model(x)
      res = torch.cat([x, res0], 1)
      return res
