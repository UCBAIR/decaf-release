'''The MNIST dataset 
'''
from decaf.layers.data import ndarraydata
import numpy as np
import os

class MNISTDataLayer(ndarraydata.NdarrayDataLayer):
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    IMAGE_DIM = (28,28)
    
    def __init__(self, **kwargs):
        """Initialize the mnist dataset.
        
        kwargs:
            is_training: whether to load the training data. Default True.
            rootfolder: the folder that stores the mnist data.
            dtype: the data type. Default numpy.float64.
        """
        is_training = kwargs.get('is_training', True)
        rootfolder = kwargs['rootfolder']
        dtype = kwargs.get('dtype', np.float64)
        self._load_mnist(rootfolder, is_training, dtype)
        # normalize data.
        self._data /= 255.
        ndarraydata.NdarrayDataLayer.__init__(
            self, sources=[self._data, self._label], **kwargs)

    def _load_mnist(self, rootfolder, is_training, dtype):
        if is_training:
            self._data = self._read_byte_data(
                    os.path.join(rootfolder,'train-images-idx3-ubyte'), 
                    16, (MNISTDataLayer.NUM_TRAIN,) + \
                            MNISTDataLayer.IMAGE_DIM).astype(dtype)
            self._label = self._read_byte_data(
                    os.path.join(rootfolder,'train-labels-idx1-ubyte'),
                    8, [MNISTDataLayer.NUM_TRAIN]).astype(np.int)
        else:
            self._data = self._read_byte_data(
                    os.path.join(rootfolder,'t10k-images-idx3-ubyte'),
                    16, (MNISTDataLayer.NUM_TEST,) + \
                            MNISTDataLayer.IMAGE_DIM).astype(dtype)
            self._label = self._read_byte_data(
                    os.path.join(rootfolder,'t10k-labels-idx1-ubyte'),
                    8, [MNISTDataLayer.NUM_TEST]).astype(np.int)
        # In the end, we will make the data 4-dimensional (num * 28 * 28 * 1)
        self._data.resize(self._data.shape + (1,))

    def _read_byte_data(self, filename, skipbytes, shape):
        fid = open(filename, 'rb')
        fid.seek(skipbytes)
        nbytes = np.prod(shape)
        data = np.fromfile(fid, dtype=np.uint8, count=nbytes)
        data.resize(shape)
        return data
