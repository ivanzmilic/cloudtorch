import numpy as np
import torch
import astropy.io.fits as fits
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# ====================================================================
def writefits(name, d):
    """
    Writes a NumPy array or image data to a FITS file.
    """
    io = fits.PrimaryHDU(d)
    io.writeto(name, overwrite=True)


# ====================================================================
def add_colorbar(im, aspect=20, pad_fraction=0.5, nbins=5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    from matplotlib import ticker
    cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    tick_locator = ticker.MaxNLocator(nbins)
    cb.locator = tick_locator
    cb.update_ticks()
    return cb



# ====================================================================
def regu(out, ii, img):
    """
    Applies a spatial regularization to the output parameters with the
    absolute difference.
    """
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[1]))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(mout[:,:,:,ii]-output_smooth))

# ====================================================================
def regu2(out, ii,img):
    """
    Applies a spatial regularization to the output parameters with the
    squared difference.
    """
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    weights = weights.to(img.device)
    mout = torch.reshape(out, (1,img.shape[0],img.shape[1],out.shape[-1]))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(output_smooth-mout[:,:,:,ii])**2.0)


# ====================================================================
def regu_mean(out, ii,img):
    """
    Applies a regularization based on the mean of the output parameters.
    """
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[1]))
    return torch.sum(torch.abs(mout[:,:,:,ii]-mout[:,:,:,ii].mean()))


# ====================================================================
def regu_min(out, ii,img, value):
    """
    Applies a regularization that penalizes values below a threshold.
    """
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[1]))
    return torch.sum(F.relu(-(mout[:,:,:,ii]-value)))

# ====================================================================
def regu_max(out, ii,img, value):
    """
    Applies a regularization that penalizes values above a threshold.
    """
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[1]))
    return torch.sum(F.relu((mout[:,:,:,ii]-value)))


# ====================================================================
def regu_value(out, ii,img, value):
    """
    Applies a regularization that penalizes deviations from a specific value.
    """
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[1]))
    return torch.sum(torch.abs(mout[:,:,:,ii]-value))


