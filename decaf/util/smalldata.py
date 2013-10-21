"""Simple utility functions to load small data for demo purpose."""

from decaf import base
import os
from skimage import io
import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(__file__), '_data')

def lena():
    """A simple function to return the lena image."""
    filename = os.path.join(_DATA_PATH, 'lena.png')
    return io.imread(filename)

def whitened_images(dtype=np.float64):
    """Returns the whitened images provided in the Sparsenet website:
        http://redwood.berkeley.edu/bruno/sparsenet/
    The returned data will be in the shape (10,512,512,1) to fit
    the blob convension.
    """
    npzdata = np.load(os.path.join(_DATA_PATH, 'whitened_images.npz'))
    blob = base.Blob(npzdata['images'].shape, dtype)
    blob.data().flat = npzdata['images'].flat
    return blob
