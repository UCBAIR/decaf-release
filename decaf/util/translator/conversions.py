"""Conversions converts data from the cuda convnet convention to the decaf
convention."""
import numpy as np



def imgs_cudaconv_to_decaf(imgs, size, channels, out=None):
    """Converting a set of images from the cudaconv order (channels first) to
    our order (channels last).
    
    Input:
        imgs: the input image. Should be in the shape
            (num, channels, size, size), but the last 3 dims could be
            flattened as long as the channels come first.
        size: the image size. Input images should be square.
        channels: the number of channels.
        out: (optional) the output matrix.
    """
    if out is None:
        out = np.empty((imgs.shape[0], size, size, channels), imgs.dtype)
    img_view = imgs.view()
    img_view.shape = (imgs.shape[0], channels, size, size)
    for i in range(channels):
        out[:, :, :, i] = img_view[:, i, :, :]
    return out

def img_cudaconv_to_decaf(img, size, channels, out=None):
    """See imgs_cudaconv_to_decaf for details. The only difference is that
    this function deals with a single image of shape (channels, size, size).
    """
    if out is None:
        out = np.empty((size, size, channels), img.dtype)
    return imgs_cudaconv_to_decaf(img[np.newaxis, :], size, channels,
                                  out=out[np.newaxis, :])[0]
